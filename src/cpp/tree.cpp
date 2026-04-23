#include<cstdint>
#include<vector>
#include<cstring>

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>

#include "tree.hpp"

namespace py = pybind11;

Tree::Tree(
    const double* X,
    int32_t n,
    int32_t p,
    const std::vector<std::vector<int32_t>>& root_rows_by_var)
    :
    X_(X),
    n_(n), 
    p_(p), 
    membership_stamp_(static_cast<size_t>(n), 0),
    stamp_id_(0),
    alive_() {

    if (X_ == nullptr) { throw std::runtime_error("X must not be null."); }
    if (n_ <= 0) { throw std::runtime_error("n must be positive."); }
    if (p_ <= 0) { throw std::runtime_error("p must be positive."); }
    if (static_cast<int32_t>(root_rows_by_var.size()) != p_) {
        throw std::runtime_error("root_rows_by_var must have length p.");
    }

    Node root;
    root.variable = -1;
    root.value = 0.0f;
    root.mu = 0.0f;
    root.left = -1;
    root.right = -1;
    root.parent = -1;
    root.split_idx = -1;
    root.rows_by_var = root_rows_by_var;
    root.rows = root.rows_by_var[0];
    root.eta_by_var.assign(static_cast<size_t>(p_), 0);

    build_node_cache(root);

    root_ = make_node(std::move(root));
    structure_cache_dirty_ = true;
}

int32_t Tree::make_node(Node&& nd) {
    if (!free_list_.empty()) {
        int32_t reused_idx = free_list_.back();
        free_list_.pop_back();

        nodes_[static_cast<size_t>(reused_idx)] = std::move(nd);
        alive_[static_cast<size_t>(reused_idx)] = 1;
        return reused_idx;
    }

    nodes_.push_back(std::move(nd));
    alive_.push_back(1);
    return static_cast<int32_t>(nodes_.size() - 1);
}

Node& Tree::node(int32_t idx) {
    if (idx < 0 || idx >= static_cast<int32_t>(nodes_.size())) {
        throw std::runtime_error("Node index out of bounds.");
    }
    if (!alive_[static_cast<size_t>(idx)]) {
        throw std::runtime_error("Access to dead node.");
    }
    return nodes_[static_cast<size_t>(idx)];
}

const Node& Tree::node(int32_t idx) const {
    if (idx < 0 || idx >= static_cast<int32_t>(nodes_.size())) {
        throw std::runtime_error("Node index out of bounds.");
    }
    if (!alive_[static_cast<size_t>(idx)]) {
        throw std::runtime_error("Access to dead node.");
    }
    return nodes_[static_cast<size_t>(idx)];
}

void Tree::build_node_cache(Node& node) {
    node.valid_vars.clear();
    node.eta_by_var.assign(static_cast<size_t>(p_), 0);

    for (int32_t var = 0; var < p_; ++var) {
        const auto& ord_v = node.rows_by_var[static_cast<size_t>(var)];
        if (ord_v.size() <= 1) continue;

        int32_t eta = 0;
        double prev = X_[static_cast<size_t>(ord_v[0]) * p_ + var];

        for (size_t k = 1; k < ord_v.size(); ++k) {
            double cur = X_[static_cast<size_t>(ord_v[k]) * p_ + var];
            if (cur != prev) {
                ++eta;
                prev = cur;
            }
        }

        if (eta > 0) {
            node.eta_by_var[static_cast<size_t>(var)] = eta;
            node.valid_vars.push_back(var);
        }
    }
}

void Tree::build_workspace_cache(WorkspaceNodeState& ws_node) const {
    ws_node.valid_vars.clear();
    ws_node.eta_by_var.assign(static_cast<size_t>(p_), 0);

    for (int32_t var = 0; var < p_; ++var) {
        const auto& ord_v = ws_node.rows_by_var[static_cast<size_t>(var)];
        if (ord_v.size() <= 1) continue;

        int32_t eta = 0;
        double prev = X_[static_cast<size_t>(ord_v[0]) * p_ + var];

        for (size_t k = 1; k < ord_v.size(); ++k) {
            double cur = X_[static_cast<size_t>(ord_v[k]) * p_ + var];
            if (cur != prev) {
                ++eta;
                prev = cur;
            }
        }

        if (eta > 0) {
            ws_node.eta_by_var[static_cast<size_t>(var)] = eta;
            ws_node.valid_vars.push_back(var);
        }
    }
}

void Tree::collect_subtree_indices(
    int32_t root_idx,
    std::vector<int32_t>& out
) const {
    std::vector<int32_t> stack{root_idx};

    while (!stack.empty()) {
        int32_t idx = stack.back();
        stack.pop_back();

        out.push_back(idx);

        const Node& cur = node(idx);
        if (cur.is_internal()) {
            stack.push_back(cur.right);
            stack.push_back(cur.left);
        }
    }
}

void Tree::collect_structure_cache(int32_t idx) const {
    const Node& cur = node(idx);

    if (cur.is_terminal()) {
        terminal_nodes_all_cache_.push_back(idx);

        if (!cur.valid_vars.empty()) {
            terminal_nodes_growable_cache_.push_back(idx);
        }
        return;
    }

    internal_nodes_cache_.push_back(idx);

    const Node& left_child = node(cur.left);
    const Node& right_child = node(cur.right);
    if (left_child.is_terminal() && right_child.is_terminal()) {
        prunable_nodes_cache_.push_back(idx);
    }
    if (left_child.is_internal() || right_child.is_internal()) {
        swappable_nodes_cache_.push_back(idx);
    }

    collect_structure_cache(cur.left);
    collect_structure_cache(cur.right);
}

int32_t Tree::count_nodes() const {
    int32_t n = 0;
    std::vector<int32_t> stack{root_};

    while (!stack.empty()) {
        int32_t idx = stack.back();
        stack.pop_back();
        ++n;

        const Node& cur = node(idx);
        if (cur.is_internal()) {
            stack.push_back(cur.right);
            stack.push_back(cur.left);
        }
    }
    return n;
}

void Tree::rebuild_structure_cache() const {
    terminal_nodes_all_cache_.clear();
    terminal_nodes_growable_cache_.clear();
    internal_nodes_cache_.clear();
    prunable_nodes_cache_.clear();
    swappable_nodes_cache_.clear();

    collect_structure_cache(root_);

    structure_cache_dirty_ = false;
}

const std::vector<int32_t>& Tree::terminal_nodes(bool growable) const {
    if (structure_cache_dirty_) {
        rebuild_structure_cache();
    }
    return growable ? terminal_nodes_growable_cache_
                    : terminal_nodes_all_cache_;
}

const std::vector<int32_t>& Tree::internal_nodes() const {
    if (structure_cache_dirty_) {
        rebuild_structure_cache();
    }
    return internal_nodes_cache_;
}

const std::vector<int32_t>& Tree::prunable_nodes() const {
    if (structure_cache_dirty_) {
        rebuild_structure_cache();
    }
    return prunable_nodes_cache_;
}

const std::vector<int32_t>& Tree::swappable_nodes() const {
    if (structure_cache_dirty_) {
        rebuild_structure_cache();
    }
    return swappable_nodes_cache_;
}

double Tree::split_value_at(
    const std::vector<int32_t>& ord_v,
    int32_t variable,
    int32_t split_idx
) const {
    if (ord_v.size() <= 1) {
        throw std::runtime_error("split_value_at called on node with <= 1 row.");
    }

    int32_t seen = -1;
    double prev = X_[static_cast<size_t>(ord_v[0]) * p_ + variable];

    for (size_t k = 1; k < ord_v.size(); ++k) {
        double cur = X_[static_cast<size_t>(ord_v[k]) * p_ + variable];
        if (cur != prev) {
            ++seen;
            if (seen == split_idx) {
                return static_cast<double>(prev);
            }
            prev = cur;
        }
    }
    throw std::runtime_error("split_idx out of range for split_value_at.");
}

bool Tree::value_present_and_splittable(const Node& cur) const {
    const auto& ord_v = cur.rows_by_var[static_cast<size_t>(cur.variable)];
    if (ord_v.size() <= 1) { return false; }

    int32_t last = -1;
    for (int32_t i = 0; i < static_cast<int32_t>(ord_v.size()); ++i) {
        double x = X_[static_cast<size_t>(ord_v[static_cast<size_t>(i)]) * p_ + cur.variable];
        if (x == cur.value) last = i;
    }
    if (last < 0) { return false; }
    if (last + 1 >= static_cast<int32_t>(ord_v.size())) { return false; }

    double x_last = X_[static_cast<size_t>(ord_v[static_cast<size_t>(last)]) * p_ + cur.variable];
    double x_next = X_[static_cast<size_t>(ord_v[static_cast<size_t>(last + 1)]) * p_ + cur.variable];
    return x_last != x_next;
}

bool Tree::partition_rows_by_var(
    const std::vector<std::vector<int32_t>>& rows_by_var,
    int32_t variable,
    const double value,
    std::vector<std::vector<int32_t>>& left_by_var,
    std::vector<std::vector<int32_t>>& right_by_var
) const {
    const auto& ord_split = rows_by_var[static_cast<size_t>(variable)];
    const int32_t n_node = static_cast<int32_t>(ord_split.size());

    if (n_node <= 1) {
        return false;
    }

    ++stamp_id_;
    if (stamp_id_ == 0) {
        std::fill(membership_stamp_.begin(), membership_stamp_.end(), 0);
        stamp_id_ = 1;
    }
    int32_t left_count = 0;
    for (int32_t row : ord_split) {
        const double x = X_[static_cast<size_t>(row) * p_ + variable];
        if (x <= value) {
            membership_stamp_[static_cast<size_t>(row)] = stamp_id_;
            ++left_count;
        }
    }

    if (left_count == 0 || left_count == n_node) {
        return false;
    }
    const int32_t right_count = n_node - left_count;

    left_by_var.clear();
    right_by_var.clear();
    left_by_var.resize(static_cast<size_t>(p_));
    right_by_var.resize(static_cast<size_t>(p_));
    for (int32_t v = 0; v < p_; ++v) {
        const auto& ord_v = rows_by_var[static_cast<size_t>(v)];
        auto& left_v = left_by_var[static_cast<size_t>(v)];
        auto& right_v = right_by_var[static_cast<size_t>(v)];

        left_v.clear();
        right_v.clear();
        left_v.reserve(static_cast<size_t>(left_count));
        right_v.reserve(static_cast<size_t>(right_count));

        for (int32_t row : ord_v) {
            if (membership_stamp_[static_cast<size_t>(row)] == stamp_id_) {
                left_v.push_back(row);
            } else {
                right_v.push_back(row);
            }
        }
    }
    return true;
}

std::optional<GrowProposalLite> Tree::propose_grow(
    int32_t node_idx,
    int32_t variable,
    int32_t split_idx
) const {
    const Node& old = node(node_idx);
    if (old.is_internal()) {
        throw std::runtime_error("Cannot grow an internal node.");
    }

    const int32_t eta = old.eta_by_var.at(static_cast<size_t>(variable));
    if (eta <= 0 || split_idx < 0 || split_idx >= eta) {
        throw std::runtime_error("Split is not valid for this node.");
    }

    const double split_value =
        split_value_at(old.rows_by_var.at(static_cast<size_t>(variable)),
                       variable,
                       split_idx);

    std::vector<std::vector<int32_t>> left_by_var, right_by_var;
    if (!partition_rows_by_var(old.rows_by_var,
                               variable,
                               split_value,
                               left_by_var,
                               right_by_var)){
        return std::nullopt;
    }

    return GrowProposalLite{
        node_idx,
        variable,
        split_idx,
        split_value,
        std::move(left_by_var),
        std::move(right_by_var)
    };
}

void Tree::apply_grow(const GrowProposalLite& proposal) {
    const int32_t cur_idx = proposal.node_idx;

    Node left_child;
    left_child.variable = -1;
    left_child.value = 0.0;
    left_child.mu = 0.0;
    left_child.left = -1;
    left_child.right = -1;
    left_child.parent = cur_idx;
    left_child.split_idx = -1;
    left_child.rows_by_var = proposal.left_by_var;
    left_child.rows = left_child.rows_by_var[0];
    left_child.eta_by_var.assign(static_cast<size_t>(p_), 0);
    build_node_cache(left_child);

    Node right_child;
    right_child.variable = -1;
    right_child.value = 0.0;
    right_child.mu = 0.0;
    right_child.left = -1;
    right_child.right = -1;
    right_child.parent = cur_idx;
    right_child.split_idx = -1;
    right_child.rows_by_var = proposal.right_by_var;
    right_child.rows = right_child.rows_by_var[0];
    right_child.eta_by_var.assign(static_cast<size_t>(p_), 0);
    build_node_cache(right_child);

    const int32_t left_idx = make_node(std::move(left_child));
    const int32_t right_idx = make_node(std::move(right_child));

    Node& cur = node(cur_idx);
    cur.variable = proposal.variable;
    cur.value = proposal.split_value;
    cur.mu = 0.0;
    cur.left = left_idx;
    cur.right = right_idx;
    cur.split_idx = proposal.split_idx;

    structure_cache_dirty_ = true;
}

std::optional<PruneProposalLite> Tree::propose_prune(
    int32_t node_idx,
    double mu
) const {
    const Node& old = node(node_idx);

    if (old.is_terminal()) {
        throw std::runtime_error("Cannot prune a terminal node.");
    }

    return PruneProposalLite{node_idx, mu};
}

void Tree::apply_prune(const PruneProposalLite& proposal) {
    Node& cur = node(proposal.node_idx);
    if (cur.is_terminal()) {
        throw std::runtime_error("apply_prune called on terminal node.");
    }

    retire_subtree(cur.left);
    retire_subtree(cur.right);

    cur.variable = -1;
    cur.value = 0.0;
    cur.mu = proposal.mu;
    cur.left = -1;
    cur.right = -1;
    cur.split_idx = -1;
    build_node_cache(cur);

    structure_cache_dirty_ = true;
}

std::optional<ChangeProposalLite> Tree::propose_change(
    int32_t node_idx,
    int32_t variable,
    int32_t split_idx
) const {
    const Node& node = this->node(node_idx);
    if (node.is_terminal()) {
        throw std::runtime_error("Cannot change a terminal node.");
    }

    const int32_t eta = node.eta_by_var.at(static_cast<size_t>(variable));
    if (eta <= 0 || split_idx < 0 || split_idx >= eta) {
        throw std::runtime_error("Split is not valid for this node.");
    }

    const double split_value =
        split_value_at(node.rows_by_var.at(static_cast<size_t>(variable)),
                       variable,
                       split_idx);

    std::vector<RuleOverride> overrides{
        RuleOverride{node_idx, variable, split_idx, split_value}
    };
    auto ws_opt = evaluate_same_shape_workspace(node_idx, overrides);
    if (!ws_opt.has_value()) {
        return std::nullopt;
    }

    return ChangeProposalLite{
        node_idx,
        variable,
        split_idx,
        split_value,
        std::move(*ws_opt)
    };
}

void Tree::apply_change(const ChangeProposalLite& proposal) {
    apply_same_shape_workspace(proposal.workspace);
}

std::optional<SwapProposalLite> Tree::propose_swap(
    int32_t node_idx,
    int mode
) const {
    const Node& parent = node(node_idx);
    if (parent.is_terminal()) {
        throw std::runtime_error("Cannot swap at a terminal node.");
    }

    const Node& left_child = node(parent.left);
    const Node& right_child = node(parent.right);

    const bool left_internal = left_child.is_internal();
    const bool right_internal = right_child.is_internal();

    if (!left_internal && !right_internal) {
        throw std::runtime_error("Cannot swap at a node with two terminal children.");
    }

    std::vector<RuleOverride> overrides;

    if (mode == SWAP_LEFT) {
        if (!left_internal) {
            throw std::runtime_error("SWAP_LEFT requested but left child is terminal.");
        }

        overrides.push_back(
            RuleOverride{
                node_idx,
                left_child.variable,
                left_child.split_idx,
                left_child.value
            }
        );
        overrides.push_back(
            RuleOverride{
                parent.left,
                parent.variable,
                parent.split_idx,
                parent.value
            }
        );
    }
    else if (mode == SWAP_RIGHT) {
        if (!right_internal) {
            throw std::runtime_error("SWAP_RIGHT requested but right child is terminal.");
        }

        overrides.push_back(
            RuleOverride{
                node_idx,
                right_child.variable,
                right_child.split_idx,
                right_child.value
            }
        );
        overrides.push_back(
            RuleOverride{
                parent.right,
                parent.variable,
                parent.split_idx,
                parent.value
            }
        );
    }
    else if (mode == SWAP_BOTH) {
        if (!left_internal || !right_internal) {
            throw std::runtime_error("SWAP_BOTH requested but both children are not internal.");
        }

        const bool same_rule =
            (left_child.variable == right_child.variable) &&
            (left_child.split_idx == right_child.split_idx) &&
            (left_child.value == right_child.value);

        if (!same_rule) {
            throw std::runtime_error("SWAP_BOTH requires both children to share the same rule.");
        }

        overrides.push_back(
            RuleOverride{
                node_idx,
                left_child.variable,
                left_child.split_idx,
                left_child.value
            }
        );
        overrides.push_back(
            RuleOverride{
                parent.left,
                parent.variable,
                parent.split_idx,
                parent.value
            }
        );
        overrides.push_back(
            RuleOverride{
                parent.right,
                parent.variable,
                parent.split_idx,
                parent.value
            }
        );
    }
    else {
        throw std::runtime_error("Unknown swap mode.");
    }

    auto ws_opt = evaluate_same_shape_workspace(node_idx, overrides);
    if (!ws_opt.has_value()) {
        return std::nullopt;
    }

    return SwapProposalLite{
        node_idx,
        mode,
        std::move(overrides),
        std::move(*ws_opt)
    };
}

void Tree::apply_swap(const SwapProposalLite& proposal) {
    apply_same_shape_workspace(proposal.workspace);
}

void Tree::apply_rebuilt_subtree_same_shape(int32_t node_idx, const Tree& rebuilt) {
    overwrite_subtree_same_shape(node_idx, rebuilt, rebuilt.root());
    structure_cache_dirty_ = true;
}

void Tree::overwrite_subtree_same_shape(
    int32_t live_idx,
    const Tree& rebuilt,
    int32_t rebuilt_idx
) {
    Node& live = node(live_idx);
    const Node& src = rebuilt.node(rebuilt_idx);

    const bool live_terminal = live.is_terminal();
    const bool src_terminal = src.is_terminal();

    if (live_terminal != src_terminal) {
        throw std::runtime_error("overwrite_subtree_same_shape: subtree shapes do not match.");
    }

    // Copy state fields only. Do NOT copy parent/left/right indices.
    live.variable = src.variable;
    live.value = src.value;
    live.mu = src.mu;
    live.split_idx = src.split_idx;
    live.rows = src.rows;
    live.rows_by_var = src.rows_by_var;
    live.valid_vars = src.valid_vars;
    live.eta_by_var = src.eta_by_var;

    if (src_terminal) {
        // Topology is already correct in the live tree.
        // These should already be -1 for a terminal node if shapes match.
        return;
    }

    overwrite_subtree_same_shape(live.left, rebuilt, src.left);
    overwrite_subtree_same_shape(live.right, rebuilt, src.right);
}

void Tree::retire_subtree(int32_t root_idx) {
    std::vector<int32_t> doomed;
    collect_subtree_indices(root_idx, doomed);

    for (int32_t idx : doomed) {
        if (!alive_[static_cast<size_t>(idx)]) {
            throw std::runtime_error("Attempted to retire an already-dead node.");
        }
        alive_[static_cast<size_t>(idx)] = 0;

        Node& nd = nodes_[static_cast<size_t>(idx)];
        nd.variable = -1;
        nd.value = 0.0f;
        nd.mu = 0.0f;
        nd.left = -1;
        nd.right = -1;
        nd.parent = -1;
        nd.split_idx = -1;
        nd.rows.clear();
        nd.rows_by_var.clear();
        nd.valid_vars.clear();
        nd.eta_by_var.clear();

        free_list_.push_back(idx);
    }
}

std::optional<SameShapeWorkspace> Tree::evaluate_same_shape_workspace(
    int32_t root_idx,
    const std::vector<RuleOverride>& overrides
) const {
    SameShapeWorkspace ws;
    ws.root_idx = root_idx;

    const Node& root = node(root_idx);

    if (!build_same_shape_workspace_dfs(
            root_idx,
            root.rows_by_var,
            overrides,
            ws)) {
        return std::nullopt;
    }

    return ws;
}

void Tree::apply_same_shape_workspace(const SameShapeWorkspace& ws) {
    for (const auto& src : ws.states) {
        Node& live = node(src.live_idx);

        live.variable = src.variable;
        live.split_idx = src.split_idx;
        live.value = src.value;
        live.rows_by_var = src.rows_by_var;
        live.rows = src.rows;
        live.valid_vars = src.valid_vars;
        live.eta_by_var = src.eta_by_var;
    }

    structure_cache_dirty_ = true;
}

bool Tree::build_same_shape_workspace_dfs(
    int32_t live_idx,
    const std::vector<std::vector<int32_t>>& rows_by_var,
    const std::vector<RuleOverride>& overrides,
    SameShapeWorkspace& ws
) const {
    const Node& live = node(live_idx);

    WorkspaceNodeState cur;
    cur.live_idx = live_idx;
    cur.rows_by_var = rows_by_var;
    cur.rows = cur.rows_by_var[0];

    if (live.is_terminal()) {
        cur.variable = -1;
        cur.split_idx = -1;
        cur.value = 0.0;
        build_workspace_cache(cur);
        ws.states.push_back(std::move(cur));
        return true;
    }
    build_workspace_cache(cur);

    const RuleOverride* ov = find_override(overrides, live_idx);
    const int32_t variable = ov ? ov->variable : live.variable;
    const int32_t split_idx = ov ? ov->split_idx : live.split_idx;
    const double split_value = ov ? ov->split_value : live.value;

    const int32_t eta = cur.eta_by_var.at(static_cast<size_t>(variable));
    if (eta <= 0 || split_idx < 0 || split_idx >= eta) {
        return false;
    }

    cur.variable = variable;
    cur.split_idx = split_idx;
    cur.value = split_value;
    ws.states.push_back(cur);

    std::vector<std::vector<int32_t>> left_by_var, right_by_var;
    if (!partition_rows_by_var(
            rows_by_var,
            variable,
            split_value,
            left_by_var,
            right_by_var)) {
        return false;
    }

    if (!build_same_shape_workspace_dfs(live.left, left_by_var, overrides, ws)) {
        return false;
    }
    if (!build_same_shape_workspace_dfs(live.right, right_by_var, overrides, ws)) {
        return false;
    }

    return true;
}

const RuleOverride* Tree::find_override(
    const std::vector<RuleOverride>& overrides,
    int32_t node_idx
) const {
    for (const auto& ov : overrides) {
        if (ov.node_idx == node_idx) return &ov;
    }
    return nullptr;
}

void Tree::serialize(std::vector<int32_t>& variable,
                     std::vector<double>& value,
                     std::vector<int32_t>& left,
                     std::vector<int32_t>& right,
                     std::vector<double>& mu) const {
    const int32_t n_nodes = count_nodes();

    variable.assign(static_cast<size_t>(n_nodes), -1);
    value.assign(static_cast<size_t>(n_nodes), 0.0f);
    left.assign(static_cast<size_t>(n_nodes), -1);
    right.assign(static_cast<size_t>(n_nodes), -1);
    mu.assign(static_cast<size_t>(n_nodes), 0.0f);

    struct Frame {
        int32_t node_idx;
        int32_t parent_ser;
        bool is_right_child;
    };

    std::vector<Frame> stack;
    stack.push_back({root_, -1, false});

    int32_t next_ser = 0;

    while (!stack.empty()) {
        Frame fr = stack.back();
        stack.pop_back();

        int32_t ser_idx = next_ser++;
        const Node& cur = node(fr.node_idx);

        if (fr.parent_ser != -1) {
            if (fr.is_right_child) right[static_cast<size_t>(fr.parent_ser)] = ser_idx;
            else left[static_cast<size_t>(fr.parent_ser)] = ser_idx;
        }

        if (cur.is_terminal()) {
            mu[static_cast<size_t>(ser_idx)] = cur.mu;
        } else {
            variable[static_cast<size_t>(ser_idx)] = cur.variable;
            value[static_cast<size_t>(ser_idx)] = cur.value;

            stack.push_back({cur.right, ser_idx, true});
            stack.push_back({cur.left, ser_idx, false});
        }
    }
}

void Tree::validate() const {
    if (root_ < 0 || root_ >= static_cast<int32_t>(nodes_.size())) {
        throw std::runtime_error("Invalid root index.");
    }

    std::vector<uint8_t> seen(nodes_.size(), 0);
    std::vector<int32_t> stack{root_};

    while (!stack.empty()) {
        int32_t idx = stack.back();
        stack.pop_back();

        if (idx < 0 || idx >= static_cast<int32_t>(nodes_.size())) {
            throw std::runtime_error("Node index out of bounds.");
        }
        if (seen[static_cast<size_t>(idx)]) {
            throw std::runtime_error("Cycle or repeated node detected.");
        }
        seen[static_cast<size_t>(idx)] = 1;

        const Node& cur = node(idx);

        if (cur.is_terminal()) {
            if (cur.right != -1) {
                throw std::runtime_error("Leaf has right child but no left child.");
            }
            if (cur.variable != -1) {
                throw std::runtime_error("Leaf has split variable.");
            }
        } else {
            if (cur.left < 0 || cur.right < 0) {
                throw std::runtime_error("Internal node missing child.");
            }
            if (cur.variable < 0 || cur.variable >= p_) {
                throw std::runtime_error("Internal node has invalid split variable.");
            }
            if (node(cur.left).parent != idx || node(cur.right).parent != idx) {
                throw std::runtime_error("Broken parent pointer.");
            }
            stack.push_back(cur.right);
            stack.push_back(cur.left);
        }

        if (static_cast<int32_t>(cur.rows_by_var.size()) != p_) {
            throw std::runtime_error("rows_by_var must have length p.");
        }
        if (cur.rows != cur.rows_by_var[0]) {
            throw std::runtime_error("rows must equal rows_by_var[0].");
        }
        if (static_cast<int32_t>(cur.eta_by_var.size()) != p_) {
            throw std::runtime_error("eta_by_var must have length p.");
        }
    }
    for (int32_t idx : free_list_) {
        if (alive_[static_cast<size_t>(idx)]) {
            throw std::runtime_error("Free list contains a live node.");
        }
    }
    if (alive_.size() != nodes_.size()) {
        throw std::runtime_error("alive_ and nodes_ size mismatch.");
    }
    std::vector<uint8_t> in_free(nodes_.size(), 0);
    for (int32_t idx : free_list_) {
        if (idx < 0 || idx >= static_cast<int32_t>(nodes_.size())) {
            throw std::runtime_error("free_list_ index out of bounds.");
        }
        if (in_free[static_cast<size_t>(idx)]) {
            throw std::runtime_error("Duplicate index in free_list_.");
        }
        in_free[static_cast<size_t>(idx)] = 1;
        if (alive_[static_cast<size_t>(idx)]) {
            throw std::runtime_error("Free list contains a live node.");
        }
    }
}

void bind_tree(py::module_& m) {
    py::class_<Tree>(m, "_Tree")
        .def(
            py::init([](py::array_t<double, py::array::c_style | py::array::forcecast> X_in) {
                if (X_in.ndim() != 2) {
                    throw std::runtime_error("X must be a 2D array.");
                }

                const int32_t n = static_cast<int32_t>(X_in.shape(0));
                const int32_t p = static_cast<int32_t>(X_in.shape(1));
                if (n <= 0 || p <= 0) {
                    throw std::runtime_error("X must have positive shape.");
                }

                auto buf = X_in.request();
                const double* X_ptr = static_cast<const double*>(buf.ptr);

                std::vector<std::vector<int32_t>> root_rows_by_var(static_cast<size_t>(p));
                for (int32_t v = 0; v < p; ++v) {
                    auto& ord = root_rows_by_var[static_cast<size_t>(v)];
                    ord.resize(static_cast<size_t>(n));
                    for (int32_t i = 0; i < n; ++i) {
                        ord[static_cast<size_t>(i)] = i;
                    }

                    std::stable_sort(ord.begin(), ord.end(),
                        [X_ptr, p, v](int32_t a, int32_t b) {
                            return X_ptr[static_cast<size_t>(a) * p + v]
                                 < X_ptr[static_cast<size_t>(b) * p + v];
                        });
                }

                return Tree(X_ptr, n, p, root_rows_by_var);
            }),
            "Create a tree with a single root node from a 2D feature matrix.",
            py::arg("X")
        )
        .def("root", &Tree::root,
            "Return the index of the root node.")
        .def("terminal_nodes",
            [](const Tree& t, bool growable) {
                const auto& v = t.terminal_nodes(growable);
                return std::vector<int32_t>(v.begin(), v.end());
            },
            "Return terminal node indices. If growable is true, return only leaves with valid splits.",
            py::arg("growable") = true)
        .def("internal_nodes",
            [](const Tree& t) {
                const auto& v = t.internal_nodes();
                return std::vector<int32_t>(v.begin(), v.end());
            },
            "Return indices of all internal nodes.")
        .def("prunable_nodes",
            [](const Tree& t) {
                const auto& v = t.prunable_nodes();
                return std::vector<int32_t>(v.begin(), v.end());
            },
            "Return internal nodes whose children are both terminal.")
        .def("swappable_nodes",
            [](const Tree& t) {
                const auto& v = t.swappable_nodes();
                return std::vector<int32_t>(v.begin(), v.end());
            },
            "Return internal nodes eligible for a swap move.")
        .def("serialize",
            [](const Tree& t) {
                std::vector<int32_t> variable, left, right;
                std::vector<double> value, mu;
                t.serialize(variable, value, left, right, mu);

                py::array_t<int32_t> variable_arr(variable.size());
                py::array_t<double> value_arr(value.size());
                py::array_t<int32_t> left_arr(left.size());
                py::array_t<int32_t> right_arr(right.size());
                py::array_t<double> mu_arr(mu.size());

                std::memcpy(variable_arr.mutable_data(), variable.data(),
                            variable.size() * sizeof(int32_t));
                std::memcpy(value_arr.mutable_data(), value.data(),
                            value.size() * sizeof(double));
                std::memcpy(left_arr.mutable_data(), left.data(),
                            left.size() * sizeof(int32_t));
                std::memcpy(right_arr.mutable_data(), right.data(),
                            right.size() * sizeof(int32_t));
                std::memcpy(mu_arr.mutable_data(), mu.data(),
                            mu.size() * sizeof(double));

                return py::make_tuple(
                    std::move(variable_arr),
                    std::move(value_arr),
                    std::move(left_arr),
                    std::move(right_arr),
                    std::move(mu_arr)
                );
            },
            "Serialize the tree into flat node arrays for variable, value, children, and leaf means.")
        .def("validate", &Tree::validate,
            "Check tree structure and cached state for consistency.")

        .def("test_grow", [](Tree& t, int32_t node_idx, int32_t variable, int32_t split_idx) {
            auto prop = t.propose_grow(node_idx, variable, split_idx);
            if (!prop.has_value()) return false;
            t.apply_grow(*prop);
            return true;
            },
            "Test function used to test proposal and application of grow moves.")
        .def("test_prune", [](Tree& t, int32_t node_idx, double mu) {
            auto prop = t.propose_prune(node_idx, mu);
            if (!prop.has_value()) return false;
            t.apply_prune(*prop);
            return true;
            },
            "Test function used to test proposal and application of prune moves.")
        .def("test_change", [](Tree& t, int32_t node_idx, int32_t variable, int32_t split_idx) {
            auto prop = t.propose_change(node_idx, variable, split_idx);
            if (!prop.has_value()) return false;
            t.apply_change(*prop);
            return true;
            },
            "Test function used to test proposal and application of change moves.")
        .def("test_swap", [](Tree& t, int32_t node_idx, int mode) {
            auto prop = t.propose_swap(node_idx, mode);
            if (!prop.has_value()) return false;
            t.apply_swap(*prop);
            return true;
            },
            "Test function used to test proposal and application of swap moves.");
}