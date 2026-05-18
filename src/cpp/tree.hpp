#pragma once

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <utility>
#include <optional>
#include <algorithm>

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

namespace py = pybind11;

struct Node {
    int32_t variable = -1;
    double value = 0.0f;
    double mu = 0.0f;

    int32_t left = -1;
    int32_t right = -1;
    int32_t parent = -1;
    int32_t split_idx = -1;

    std::vector<int32_t> rows;
    std::vector<std::vector<int32_t>> rows_by_var;
    std::vector<int32_t> valid_vars;
    std::vector<int32_t> eta_by_var;

    bool is_terminal() const noexcept { return left == -1; }
    bool is_internal() const noexcept { return left != -1; }
};

struct GrowProposalLite {
    int32_t node_idx;
    int32_t variable;
    int32_t split_idx;
    double split_value;
    std::vector<std::vector<int32_t>> left_by_var;
    std::vector<std::vector<int32_t>> right_by_var;
};

struct PruneProposalLite {
    int32_t node_idx;
    double mu;
};

struct RuleOverride {
    int32_t node_idx;
    int32_t variable;
    int32_t split_idx;
    double split_value;
};

struct WorkspaceNodeState {
    int32_t live_idx;
    int32_t variable;
    int32_t split_idx;
    double value;

    std::vector<std::vector<int32_t>> rows_by_var;
    std::vector<int32_t> rows;
    std::vector<int32_t> valid_vars;
    std::vector<int32_t> eta_by_var;
};

struct SameShapeWorkspace {
    int32_t root_idx;
    std::vector<WorkspaceNodeState> states;
};

struct ChangeProposalLite {
    int32_t node_idx;
    int32_t variable;
    int32_t split_idx;
    double split_value;
    SameShapeWorkspace workspace;
};

static constexpr int SWAP_LEFT = 0;
static constexpr int SWAP_RIGHT = 1;
static constexpr int SWAP_BOTH = 2;

struct SwapProposalLite {
    int32_t node_idx;
    int mode;
    std::vector<RuleOverride> overrides;
    SameShapeWorkspace workspace;
};

/**
 * @brief Mutable regression-tree representation used during BART MCMC.
 *
 * Tree stores the topology, split rules, terminal-node means, row membership,
 * and split caches for one regression tree in the live ensemble. It supports
 * grow, prune, change, and swap proposals by constructing lightweight proposal
 * objects first, then applying accepted proposals in place.
 *
 * The tree keeps row indices sorted by predictor within each node so that valid
 * split rules and child row partitions can be computed efficiently.
 */
class Tree {
    public:
    Tree(
        const double* X,
        int32_t n,
        int32_t p,
        const std::vector<std::vector<int32_t>>& root_rows_by_var
    );

    int32_t root() const noexcept { return root_; }
    Node& node(int32_t idx);
    const Node& node(int32_t idx) const;

    const std::vector<int32_t>& terminal_nodes(bool growable) const;
    const std::vector<int32_t>& internal_nodes() const;
    const std::vector<int32_t>& prunable_nodes() const;
    const std::vector<int32_t>& swappable_nodes() const;
    
    double split_value_at(
        const std::vector<int32_t>& ord_v,
        int32_t variable,
        int32_t split_idx
    ) const;

    std::optional<int32_t> split_idx_of_value(
        const std::vector<int32_t>& ord_v,
        int32_t variable,
        double value
    ) const;

    std::optional<GrowProposalLite> propose_grow(
        int32_t node_idx,
        int32_t variable,
        int32_t split_idx
    ) const;
    void apply_grow(const GrowProposalLite& proposal);

    std::optional<PruneProposalLite> propose_prune(
        int32_t node_idx,
        double mu = 0.0f
    ) const;
    void apply_prune(const PruneProposalLite& proposal);

    std::optional<ChangeProposalLite> propose_change(
        int32_t node_idx,
        int32_t variable,
        int32_t split_idx
    ) const;
    void apply_change(const ChangeProposalLite& proposal);

    std::optional<SwapProposalLite> propose_swap(
        int32_t node_idx,
        int mode
    ) const;
    void apply_swap(const SwapProposalLite& proposal);
    
    int32_t count_nodes() const;
    void serialize(
        std::vector<int32_t>& variable,
        std::vector<double>& value,
        std::vector<int32_t>& left,
        std::vector<int32_t>& right,
        std::vector<double>& mu
    ) const;

    void validate() const;

private:
    const double* X_;
    int32_t n_;
    int32_t p_;
    int32_t root_;

    std::vector<Node> nodes_;
    std::vector<int32_t> free_list_;
    std::vector<uint8_t> alive_;
    mutable std::vector<int32_t> membership_stamp_;
    mutable int32_t stamp_id_ = 0;

    mutable bool structure_cache_dirty_ = true;

    mutable std::vector<int32_t> terminal_nodes_all_cache_;
    mutable std::vector<int32_t> terminal_nodes_growable_cache_;
    mutable std::vector<int32_t> internal_nodes_cache_;
    mutable std::vector<int32_t> prunable_nodes_cache_;
    mutable std::vector<int32_t> swappable_nodes_cache_;

    int32_t make_node(Node&& node);

    void build_node_cache(Node& node);
    void rebuild_structure_cache() const;
    void collect_structure_cache(int32_t idx) const;

    bool partition_rows_by_var(
        const std::vector<std::vector<int32_t>>& rows_by_var,
        int32_t variable,
        const double value,
        std::vector<std::vector<int32_t>>& left_by_var,
        std::vector<std::vector<int32_t>>& right_by_var
    ) const;
    
    void collect_subtree_indices(int32_t root_idx, std::vector<int32_t>& out) const;
    void retire_subtree(int32_t root_idx);
    void overwrite_subtree_same_shape(
        int32_t live_idx,
        const Tree& rebuilt,
        int32_t rebuilt_idx
    );

    void apply_rebuilt_subtree_same_shape(int32_t node_idx, const Tree& rebuilt);
    std::optional<SameShapeWorkspace> evaluate_same_shape_workspace(
        int32_t root_idx,
        const std::vector<RuleOverride>& overrides
    ) const;
    void apply_same_shape_workspace(const SameShapeWorkspace& ws);
    bool build_same_shape_workspace_dfs(
        int32_t live_idx,
        const std::vector<std::vector<int32_t>>& rows_by_var,
        const std::vector<RuleOverride>& overrides,
        SameShapeWorkspace& ws
    ) const;
    void build_workspace_cache(WorkspaceNodeState& ws_node) const;

    const RuleOverride* find_override(
        const std::vector<RuleOverride>& overrides,
        int32_t node_idx
    ) const;
    bool value_present_and_splittable(const Node& node) const;
};

void bind_tree(py::module_& m);