// sal's C entry point to OpenGM's inference, built by infra/build_opengm.sh (issue #1279).
//
// One call poses a pairwise Potts model in sal's sign and runs one OpenGM
// algorithm on it. The energy is sum_i unary[i, x_i] - sum_e J_e [x_a == x_b]:
// the unary is -h (sal.sim.potts.energy's field, negated), each edge a
// PottsFunction with -J on the diagonal and 0 off it, so OpenGM's bound is a
// bound on that energy, with no constant to restore. Alpha expansion and the
// swap receive the metric form instead, 0 on the diagonal and J off it: their
// auxiliary-node construction (Boykov, Veksler and Zabih 2001) assumes
// V(a, a) = 0 and is wrong otherwise, which the first form showed as
// labellings sal's expansion still lowered. The form differs by sum_e J_e, a
// constant, so the labelling is the same problem's; `value` is evaluated in
// sal's form whatever the algorithm saw.
//
// Every algorithm is OpenGM's header-only code and needs no downloaded
// external. Alpha expansion and the alpha-beta swap need a min s-t cut, which
// OpenGM takes from Boost Graph or from Kolmogorov's research-only maxflow;
// neither is used here: `Dinic` below is the cut, written against the
// interface OpenGM's GraphCut calls (addEdge, calculateCut).
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/icm.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/astar.hxx>
#include <opengm/inference/trws/trws_trws.hxx>
#include <opengm/inference/dynamicprogramming.hxx>
#include <opengm/inference/dualdecomposition/dualdecomposition_subgradient.hxx>
#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/alphaexpansion.hxx>
#include <opengm/inference/alphabetaswap.hxx>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <queue>
#include <vector>

namespace {

typedef opengm::SimpleDiscreteSpace<std::size_t, std::size_t> Space;
typedef opengm::GraphicalModel<
    double, opengm::Adder,
    OPENGM_TYPELIST_2(opengm::ExplicitFunction<double>, opengm::PottsFunction<double>),
    Space>
    Model;

// Dinic's maximum flow on double capacities, for OpenGM's GraphCut. Node 0 is
// the source and node 1 the sink; calculateCut marks a node true where it is
// not reachable from the source in the residual graph, as OpenGM's
// MinSTCutBoost does.
class Dinic {
 public:
  typedef std::size_t node_type;
  typedef double ValueType;

  Dinic(std::size_t n_nodes, std::size_t n_edges) : head_(n_nodes, -1), level_(n_nodes), cursor_(n_nodes) {
    to_.reserve(2 * n_edges);
    capacity_.reserve(2 * n_edges);
    next_.reserve(2 * n_edges);
  }

  void addEdge(node_type tail, node_type head, ValueType capacity) {
    push(tail, head, capacity);
    push(head, tail, 0.0);
  }

  void calculateCut(std::vector<bool>& segmentation) {
    while (layer()) {
      std::copy(head_.begin(), head_.end(), cursor_.begin());
      while (augment(0, std::numeric_limits<double>::infinity()) > 0.0) {
      }
    }
    layer();
    segmentation.assign(head_.size(), true);
    for (std::size_t node = 0; node < head_.size(); ++node) {
      segmentation[node] = level_[node] < 0;
    }
    segmentation[0] = false;
    segmentation[1] = false;
  }

 private:
  void push(node_type tail, node_type head, ValueType capacity) {
    to_.push_back(static_cast<long>(head));
    capacity_.push_back(capacity);
    next_.push_back(head_[tail]);
    head_[tail] = static_cast<long>(to_.size()) - 1;
  }

  bool layer() {
    std::fill(level_.begin(), level_.end(), -1);
    std::queue<long> queue;
    level_[0] = 0;
    queue.push(0);
    while (!queue.empty()) {
      long node = queue.front();
      queue.pop();
      for (long arc = head_[node]; arc >= 0; arc = next_[arc]) {
        if (capacity_[arc] > 0.0 && level_[to_[arc]] < 0) {
          level_[to_[arc]] = level_[node] + 1;
          queue.push(to_[arc]);
        }
      }
    }
    return level_[1] >= 0;
  }

  double augment(long node, double limit) {
    if (node == 1) {
      return limit;
    }
    for (long& arc = cursor_[node]; arc >= 0; arc = next_[arc]) {
      long next = to_[arc];
      if (capacity_[arc] > 0.0 && level_[next] == level_[node] + 1) {
        double sent = augment(next, std::min(limit, capacity_[arc]));
        if (sent > 0.0) {
          capacity_[arc] -= sent;
          capacity_[arc ^ 1] += sent;
          return sent;
        }
      }
    }
    return 0.0;
  }

  std::vector<long> head_, level_, cursor_, to_, next_;
  std::vector<double> capacity_;
};

// Records the bound after each iteration an algorithm reports.
template <class INF>
struct Trace {
  std::vector<double>* bounds;
  void begin(INF&) {}
  std::size_t operator()(INF& inference) {
    bounds->push_back(inference.bound());
    return opengm::visitors::VisitorReturnFlag::ContinueInf;
  }
  void end(INF&) {}
  void addLog(const std::string&) {}
  void log(const std::string&, double) {}
};

enum Algorithm { ICM = 0, LBP = 1, ASTAR = 2, TRWS = 3, DD = 4, EXPANSION = 5, SWAP = 6 };

typedef std::chrono::steady_clock Clock;

double since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

template <class INF>
void read(const INF& inference, std::vector<std::size_t>& labels, double* bound) {
  inference.arg(labels);
  *bound = inference.bound();
}

}  // namespace

extern "C" {

// Run `algorithm` on the model and write the labelling, its energy in sal's form,
// the algorithm's bound, its per-iteration bound trace (at most
// `max_iterations` entries), the entries written, and the seconds of the
// inference call alone. Returns 0, or 1 with `error` holding the message.
int sal_opengm_solve(int algorithm, std::int64_t n_nodes, std::int64_t n_states, const double* unary,
                     std::int64_t n_edges, const std::int64_t* first, const std::int64_t* second,
                     const double* coupling, const std::int64_t* start, std::int64_t max_iterations,
                     double tolerance, std::int64_t* labels_out, double* value, double* bound,
                     double* trace, std::int64_t* taken, double* seconds, char* error,
                     std::int64_t error_size) {
  try {
    Space space(static_cast<std::size_t>(n_nodes), static_cast<std::size_t>(n_states));
    Model model(space);
    const std::size_t shape[] = {static_cast<std::size_t>(n_states)};
    for (std::int64_t node = 0; node < n_nodes; ++node) {
      opengm::ExplicitFunction<double> table(shape, shape + 1);
      for (std::int64_t state = 0; state < n_states; ++state) {
        table(static_cast<std::size_t>(state)) = unary[node * n_states + state];
      }
      const std::size_t variable[] = {static_cast<std::size_t>(node)};
      model.addFactor(model.addFunction(table), variable, variable + 1);
    }
    const bool metric = algorithm == EXPANSION || algorithm == SWAP;
    for (std::int64_t edge = 0; edge < n_edges; ++edge) {
      opengm::PottsFunction<double> potts(n_states, n_states, metric ? 0.0 : -coupling[edge],
                                          metric ? coupling[edge] : 0.0);
      const std::size_t ends[] = {static_cast<std::size_t>(first[edge]),
                                  static_cast<std::size_t>(second[edge])};
      model.addFactor(model.addFunction(potts), ends, ends + 2);
    }
    std::vector<std::size_t> labels(static_cast<std::size_t>(n_nodes), 0);
    std::vector<std::size_t> initial;
    if (start != nullptr) {
      initial.assign(start, start + n_nodes);
    }
    std::vector<double> bounds;
    const std::size_t steps = static_cast<std::size_t>(max_iterations);
    Clock::time_point clock;
    *bound = -std::numeric_limits<double>::infinity();
    switch (algorithm) {
      case ICM: {
        typedef opengm::ICM<Model, opengm::Minimizer> Inference;
        typename Inference::Parameter parameter(initial);
        Inference inference(model, parameter);
        clock = Clock::now();
        inference.infer();
        *seconds = since(clock);
        read(inference, labels, bound);
        break;
      }
      case LBP: {
        typedef opengm::BeliefPropagationUpdateRules<Model, opengm::Minimizer> Rules;
        typedef opengm::MessagePassing<Model, opengm::Minimizer, Rules, opengm::MaxDistance> Inference;
        typename Inference::Parameter parameter(steps, tolerance);
        Inference inference(model, parameter);
        Trace<Inference> visitor = {&bounds};
        clock = Clock::now();
        inference.infer(visitor);
        *seconds = since(clock);
        read(inference, labels, bound);
        break;
      }
      case ASTAR: {
        typedef opengm::AStar<Model, opengm::Minimizer> Inference;
        Inference inference(model);
        clock = Clock::now();
        inference.infer();
        *seconds = since(clock);
        read(inference, labels, bound);
        break;
      }
      case TRWS: {
        typedef opengm::TRWSi<Model, opengm::Minimizer> Inference;
        typename Inference::Parameter parameter(steps);
        // sal.search.trws.trws's criterion as OpenGM states it: the bound
        // rose by at most `tolerance |bound|`, or the gap closed to that.
        parameter.precision() = tolerance;
        parameter.isAbsolutePrecision() = false;
        parameter.minRelativeDualImprovement() = tolerance;
        Inference inference(model, parameter);
        Trace<Inference> visitor = {&bounds};
        clock = Clock::now();
        inference.infer(visitor);
        *seconds = since(clock);
        read(inference, labels, bound);
        break;
      }
      case DD: {
        typedef opengm::DDDualVariableBlock<marray::Marray<double> > Block;
        typedef opengm::DualDecompositionBase<Model, Block>::SubGmType SubModel;
        typedef opengm::DynamicProgramming<SubModel, opengm::Minimizer> SubInference;
        typedef opengm::DualDecompositionSubGradient<Model, SubInference, Block> Inference;
        typename Inference::Parameter parameter;
        parameter.maximalNumberOfIterations_ = steps;
        parameter.minimalRelAccuracy_ = tolerance;
        Inference inference(model, parameter);
        Trace<Inference> visitor = {&bounds};
        clock = Clock::now();
        inference.infer(visitor);
        *seconds = since(clock);
        read(inference, labels, bound);
        break;
      }
      case EXPANSION:
      case SWAP: {
        typedef opengm::GraphCut<Model, opengm::Minimizer, Dinic> Cut;
        typename Cut::Parameter cut;
        if (algorithm == EXPANSION) {
          typedef opengm::AlphaExpansion<Model, Cut> Inference;
          typename Inference::Parameter parameter(steps, cut);
          if (start != nullptr) {
            parameter.labelInitialType_ = Inference::Parameter::EXPLICIT_LABEL;
            parameter.label_ = initial;
          }
          Inference inference(model, parameter);
          Trace<Inference> visitor = {&bounds};
          clock = Clock::now();
          inference.infer(visitor);
          *seconds = since(clock);
          read(inference, labels, bound);
        } else {
          typedef opengm::AlphaBetaSwap<Model, Cut> Inference;
          typename Inference::Parameter parameter;
          parameter.parameter_ = cut;
          parameter.maxNumberOfIterations_ = steps;
          Inference inference(model, parameter);
          if (start != nullptr) {
            inference.setStartingPoint(initial.begin());
          }
          Trace<Inference> visitor = {&bounds};
          clock = Clock::now();
          inference.infer(visitor);
          *seconds = since(clock);
          read(inference, labels, bound);
        }
        *bound = -std::numeric_limits<double>::infinity();
        break;
      }
      default:
        throw std::runtime_error("unknown algorithm");
    }
    // The labelling's energy in sal's form, nodes then edges in order.
    *value = 0.0;
    for (std::int64_t node = 0; node < n_nodes; ++node) {
      labels_out[node] = static_cast<std::int64_t>(labels[static_cast<std::size_t>(node)]);
      *value += unary[node * n_states + labels_out[node]];
    }
    for (std::int64_t edge = 0; edge < n_edges; ++edge) {
      if (labels_out[first[edge]] == labels_out[second[edge]]) {
        *value -= coupling[edge];
      }
    }
    if (algorithm == ASTAR) {
      *bound = *value;  // exact: the minimum bounds itself
    }
    const std::size_t written = std::min(bounds.size(), steps);
    std::copy(bounds.begin(), bounds.begin() + written, trace);
    *taken = static_cast<std::int64_t>(written);
    return 0;
  } catch (const std::exception& failure) {
    std::strncpy(error, failure.what(), static_cast<std::size_t>(error_size - 1));
    error[error_size - 1] = '\0';
    return 1;
  }
}

}  // extern "C"
