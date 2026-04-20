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

struct ProposalSubtree;

class Tree {
public:
    Tree(const double* X,
         int32_t n,
         int32_t p,
         const std::vector<std::vector<int32_t>>& root_rows_by_var);

         Node& node(int32_t idx);
         const Node& node(int32_t idx) const;

    int32_t root() const noexcept { return root_; }

    std::vector<int32_t> terminal_nodes(bool growable = true) const;
    std::vector<int32_t> internal_nodes() const;
    std::vector<int32_t> prunable_nodes() const;
    std::vector<int32_t> swappable_nodes() const;

    int32_t count_nodes() const;

    double split_value_at(const std::vector<int32_t>& ord_v,
                         int32_t variable,
                         int32_t split_idx) const;

    int32_t split_pos_of_value(const std::vector<int32_t>& ord_v,
                               int32_t variable,
                               double value) const;

    std::optional<ProposalSubtree> propose_grow(int32_t node_idx,
                                                int32_t variable,
                                                int32_t split_idx) const;
    std::optional<ProposalSubtree> propose_prune(int32_t node_idx,
                                                double mu = 0.0f) const;
    std::optional<ProposalSubtree> propose_change(int32_t node_idx,
                                                int32_t variable,
                                                int32_t split_idx) const;
    std::optional<ProposalSubtree> propose_swap(int32_t node_idx,
                                                int mode) const;   // 0=left, 1=right, 2=both

    void replace_subtree(int32_t node_idx, const Tree& replacement);

    void serialize(std::vector<int32_t>& variable,
                   std::vector<double>& value,
                   std::vector<int32_t>& left,
                   std::vector<int32_t>& right,
                   std::vector<double>& mu) const;

    void validate() const;

private:
    const double* X_;
    int32_t n_;
    int32_t p_;
    int32_t root_;

    std::vector<Node> nodes_;
    std::vector<int32_t> free_list_;
    std::vector<uint8_t> alive_;
    std::vector<int32_t> membership_stamp_;
    int32_t stamp_id_ = 0;

    int32_t make_node(Node&& node);

    void build_node_cache(Node& node);

    bool partition_rows_by_var(const std::vector<std::vector<int32_t>>& rows_by_var,
                               int32_t variable,
                               double value,
                               std::vector<std::vector<int32_t>>& left_by_var,
                               std::vector<std::vector<int32_t>>& right_by_var);
    
    void collect_subtree_indices(int32_t root_idx, std::vector<int32_t>& out) const;

    void collect_terminal_nodes(int32_t idx,
                                bool growable,
                                std::vector<int32_t>& out) const;
    void collect_internal_nodes(int32_t idx,
                                std::vector<int32_t>& out) const;
    void collect_prunable_nodes(int32_t idx,
                                std::vector<int32_t>& out) const;
    void collect_swappable_nodes(int32_t idx,
                                 std::vector<int32_t>& out) const;

    bool value_present_and_splittable(const Node& node) const;

    Tree copy_subtree(int32_t node_idx) const;

    void retire_subtree(int32_t root_idx);

    bool update_subtree_from_root(int32_t node_idx,
                                std::vector<std::vector<int32_t>>* terminals,
                                std::vector<int32_t>* internals);
};

struct ProposalSubtree {
    Tree subtree;
    std::vector<std::vector<int32_t>> terminals;
    std::vector<int32_t> internals;
};

void bind_tree(py::module_& m);