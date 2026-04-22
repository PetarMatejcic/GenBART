#include<cstdint>
#include<vector>
#include<cstring>

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>

#include "tree.hpp"

namespace py = pybind11;

Tree::Tree(const double* X,
           int32_t n,
           int32_t p,
           const std::vector<std::vector<int32_t>>& root_rows_by_var)
    : X_(X), n_(n), p_(p), membership_stamp_(static_cast<size_t>(n), 0), stamp_id_(0), alive_() {

    if (X_ == nullptr) throw std::runtime_error("X must not be null.");
    if (n_ <= 0) throw std::runtime_error("n must be positive.");
    if (p_ <= 0) throw std::runtime_error("p must be positive.");
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
        int32_t idx = free_list_.back();
        free_list_.pop_back();

        nodes_[static_cast<size_t>(idx)] = std::move(nd);
        alive_[static_cast<size_t>(idx)] = 1;
        return idx;
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

void Tree::collect_subtree_indices(int32_t root_idx,
                                   std::vector<int32_t>& out) const {
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

double Tree::split_value_at(const std::vector<int32_t>& ord_v,
                           int32_t variable,
                           int32_t split_idx) const {
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
                return static_cast<double>(prev);   // not cur
            }
            prev = cur;
        }
    }

    throw std::runtime_error("split_idx out of range for split_value_at.");
}

int32_t Tree::split_pos_of_value(const std::vector<int32_t>& ord_v,
                                 int32_t variable,
                                 double value) const {
    if (ord_v.empty()) {
        throw std::runtime_error("split_pos_of_value called on empty rows.");
    }

    int32_t last = -1;
    for (int32_t i = 0; i < static_cast<int32_t>(ord_v.size()); ++i) {
        double x = X_[static_cast<size_t>(ord_v[static_cast<size_t>(i)]) * p_ + variable];
        if (x == value) last = i;
    }

    if (last < 0) {
        throw std::runtime_error("Current split value is not present in node rows.");
    }
    if (last + 1 >= static_cast<int32_t>(ord_v.size())) {
        throw std::runtime_error("Current split value is not a valid split for this node.");
    }

    double x_last = X_[static_cast<size_t>(ord_v[static_cast<size_t>(last)]) * p_ + variable];
    double x_next = X_[static_cast<size_t>(ord_v[static_cast<size_t>(last + 1)]) * p_ + variable];
    if (x_last == x_next) {
        throw std::runtime_error("Current split value is not a valid split for this node.");
    }

    int32_t pos = 0;
    double prev = X_[static_cast<size_t>(ord_v[0]) * p_ + variable];
    for (int32_t i = 1; i <= last; ++i) {
        double cur = X_[static_cast<size_t>(ord_v[static_cast<size_t>(i)]) * p_ + variable];
        if (cur != prev) {
            if (i - 1 == last) break;
            ++pos;
            prev = cur;
        }
    }
    return pos;
}

bool Tree::partition_rows_by_var(const std::vector<std::vector<int32_t>>& rows_by_var,
                                 int32_t variable,
                                 const double value,
                                 std::vector<std::vector<int32_t>>& left_by_var,
                                 std::vector<std::vector<int32_t>>& right_by_var) const{
    const auto& ord_split = rows_by_var[static_cast<size_t>(variable)];

    std::vector<int32_t> left_rows;
    left_rows.reserve(ord_split.size());

    for (int32_t row : ord_split) {
        double x = X_[static_cast<size_t>(row) * p_ + variable];
        if (x <= value) left_rows.push_back(row);
    }

    if (left_rows.empty() || left_rows.size() == ord_split.size()) {
        return false;
    }

    ++stamp_id_;
    if (stamp_id_ == 0) {
        std::fill(membership_stamp_.begin(), membership_stamp_.end(), 0);
        stamp_id_ = 1;
    }

    for (int32_t row : left_rows) {
        membership_stamp_[static_cast<size_t>(row)] = stamp_id_;
    }

    left_by_var.assign(static_cast<size_t>(p_), {});
    right_by_var.assign(static_cast<size_t>(p_), {});

    for (int32_t v = 0; v < p_; ++v) {
        const auto& ord_v = rows_by_var[static_cast<size_t>(v)];
        auto& left_v = left_by_var[static_cast<size_t>(v)];
        auto& right_v = right_by_var[static_cast<size_t>(v)];
        left_v.reserve(ord_v.size());
        right_v.reserve(ord_v.size());

        for (int32_t row : ord_v) {
            if (membership_stamp_[static_cast<size_t>(row)] == stamp_id_) {
                left_v.push_back(row);
            } else {
                right_v.push_back(row);
            }
        }
    }

    return !(left_by_var[0].empty() || right_by_var[0].empty());
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

void Tree::apply_rebuilt_subtree_same_shape(int32_t node_idx, const Tree& rebuilt) {
    overwrite_subtree_same_shape(node_idx, rebuilt, rebuilt.root());
    structure_cache_dirty_ = true;
}

bool Tree::value_present_and_splittable(const Node& cur) const {
    const auto& ord_v = cur.rows_by_var[static_cast<size_t>(cur.variable)];
    if (ord_v.size() <= 1) return false;

    int32_t last = -1;
    for (int32_t i = 0; i < static_cast<int32_t>(ord_v.size()); ++i) {
        double x = X_[static_cast<size_t>(ord_v[static_cast<size_t>(i)]) * p_ + cur.variable];
        if (x == cur.value) last = i;
    }
    if (last < 0) return false;
    if (last + 1 >= static_cast<int32_t>(ord_v.size())) return false;

    double x_last = X_[static_cast<size_t>(ord_v[static_cast<size_t>(last)]) * p_ + cur.variable];
    double x_next = X_[static_cast<size_t>(ord_v[static_cast<size_t>(last + 1)]) * p_ + cur.variable];
    return x_last != x_next;
}

SubtreeSnapshot Tree::snapshot_subtree(int32_t root_idx) const {
    std::vector<int32_t> idxs;
    collect_subtree_indices(root_idx, idxs);

    SubtreeSnapshot out;
    out.root_idx = root_idx;
    out.node_indices = idxs;
    out.saved_nodes.reserve(idxs.size());

    for (int32_t idx : idxs) {
        out.saved_nodes.push_back(node(idx));
    }

    return out;
}

void Tree::restore_subtree(const SubtreeSnapshot& snapshot) {
    if (snapshot.node_indices.size() != snapshot.saved_nodes.size()) {
        throw std::runtime_error("snapshot_subtree/restore_subtree size mismatch.");
    }

    for (size_t k = 0; k < snapshot.node_indices.size(); ++k) {
        const int32_t idx = snapshot.node_indices[k];
        nodes_[static_cast<size_t>(idx)] = snapshot.saved_nodes[k];
    }
}

Tree Tree::copy_subtree(int32_t node_idx) const {
    std::vector<int32_t> old_to_new(nodes_.size(), -1);
    std::vector<int32_t> stack{node_idx};
    std::vector<int32_t> order;

    while (!stack.empty()) {
        int32_t cur = stack.back();
        stack.pop_back();
        order.push_back(cur);

        const Node& nd = node(cur);
        if (nd.is_internal()) {
            stack.push_back(nd.right);
            stack.push_back(nd.left);
        }
    }

    Tree out(X_, n_, p_, node(node_idx).rows_by_var);
    out.nodes_.clear();
    out.alive_.clear();
    out.free_list_.clear();
    out.root_ = 0;
    out.membership_stamp_.assign(static_cast<size_t>(n_), 0);
    out.stamp_id_ = 0;

    for (int32_t old_idx : order) {
        Node copied = node(old_idx);
        copied.parent = -1;
        copied.left = -1;
        copied.right = -1;

        old_to_new[static_cast<size_t>(old_idx)] = out.make_node(std::move(copied));
    }

    for (int32_t old_idx : order) {
        int32_t new_idx = old_to_new[static_cast<size_t>(old_idx)];
        const Node& old_node = node(old_idx);
        Node& new_node = out.nodes_[static_cast<size_t>(new_idx)];

        new_node.parent = (old_idx == node_idx)
            ? -1
            : old_to_new[static_cast<size_t>(old_node.parent)];

        if (old_node.is_internal()) {
            new_node.left = old_to_new[static_cast<size_t>(old_node.left)];
            new_node.right = old_to_new[static_cast<size_t>(old_node.right)];
        }
    }

    out.alive_.assign(out.nodes_.size(), 1);
    out.free_list_.clear();
    return out;
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

bool Tree::update_subtree_from_root(int32_t node_idx) {
    Node& cur = node(node_idx);
    cur.rows = cur.rows_by_var[0];

    if (cur.is_terminal()) {
        build_node_cache(cur);
        return true;
    }

    build_node_cache(cur);

    if (!value_present_and_splittable(cur)) {
        return false;
    }

    std::vector<std::vector<int32_t>> left_by_var, right_by_var;
    if (!partition_rows_by_var(cur.rows_by_var, cur.variable, cur.value, left_by_var, right_by_var)) {
        return false;
    }

    {
        Node& left_child = node(cur.left);
        left_child.rows_by_var = std::move(left_by_var);
        left_child.rows = left_child.rows_by_var[0];
    }
    {
        Node& right_child = node(cur.right);
        right_child.rows_by_var = std::move(right_by_var);
        right_child.rows = right_child.rows_by_var[0];
    }

    if (!update_subtree_from_root(cur.left)) return false;
    if (!update_subtree_from_root(cur.right)) return false;

    return true;
}

bool Tree::propose_change(int32_t node_idx,
                          int32_t variable,
                          int32_t split_idx) {
    Node& old = node(node_idx);

    if (old.is_terminal()) {
        throw std::runtime_error("Cannot change split rule of a terminal node.");
    }

    int32_t eta = old.eta_by_var.at(static_cast<size_t>(variable));
    if (eta <= 0 || split_idx < 0 || split_idx >= eta) {
        throw std::runtime_error("Split index is not valid for this node.");
    }

    old.variable = variable;
    old.value = split_value_at(
        old.rows_by_var.at(static_cast<size_t>(variable)),
        variable,
        split_idx
    );
    old.split_idx = split_idx;

    return update_subtree_from_root(node_idx);
}

bool Tree::propose_swap(int32_t node_idx,
                        int mode) {
    Node& old_parent = node(node_idx);
    if (old_parent.is_terminal()) {
        throw std::runtime_error("swap requires an internal parent.");
    }

    Tree proposal = copy_subtree(node_idx);
    Node& parent = proposal.node(proposal.root_);

    if (mode == 0) {   // left
        Node& child = proposal.node(parent.left);
        if (child.is_terminal()) return false;

        std::swap(parent.variable, child.variable);
        std::swap(parent.value, child.value);
        std::swap(parent.split_idx, child.split_idx);
    }
    else if (mode == 1) {   // right
        Node& child = proposal.node(parent.right);
        if (child.is_terminal()) return false;

        std::swap(parent.variable, child.variable);
        std::swap(parent.value, child.value);
        std::swap(parent.split_idx, child.split_idx);
    }
    else if (mode == 2) {   // both
        Node& left_child = proposal.node(parent.left);
        Node& right_child = proposal.node(parent.right);
        if (left_child.is_terminal() || right_child.is_terminal()) return false;

        const int32_t old_parent_var = parent.variable;
        const double old_parent_val = parent.value;
        const int32_t old_parent_split = parent.split_idx;

        parent.variable = left_child.variable;
        parent.value = left_child.value;
        parent.split_idx = left_child.split_idx;

        left_child.variable = old_parent_var;
        left_child.value = old_parent_val;
        left_child.split_idx = old_parent_split;

        right_child.variable = old_parent_var;
        right_child.value = old_parent_val;
        right_child.split_idx = old_parent_split;
    }
    else {
        throw std::runtime_error("swap mode must be 0, 1, or 2.");
    }

    return update_subtree_from_root(node_idx);
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
            py::arg("X")
        )
        .def("root", &Tree::root)
        .def("terminal_nodes",
            [](const Tree& t, bool growable) {
                const auto& v = t.terminal_nodes(growable);
                return std::vector<int32_t>(v.begin(), v.end());
            },
            py::arg("growable") = true)
        .def("internal_nodes",
            [](const Tree& t) {
                const auto& v = t.internal_nodes();
                return std::vector<int32_t>(v.begin(), v.end());
            })
        .def("prunable_nodes",
            [](const Tree& t) {
                const auto& v = t.prunable_nodes();
                return std::vector<int32_t>(v.begin(), v.end());
            })
        .def("swappable_nodes",
            [](const Tree& t) {
                const auto& v = t.swappable_nodes();
                return std::vector<int32_t>(v.begin(), v.end());
            })
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
            }
        )
        .def("validate", &Tree::validate)
        .def("propose_grow", &Tree::propose_grow,
             py::arg("node_idx"), py::arg("variable"), py::arg("split_idx"))
        .def("propose_prune", &Tree::propose_prune,
             py::arg("node_idx"), py::arg("mu") = 0.0f)
        .def("propose_change", &Tree::propose_change,
             py::arg("node_idx"), py::arg("variable"), py::arg("split_idx"))
        .def("propose_swap", &Tree::propose_swap,
             py::arg("node_idx"), py::arg("mode"));
}