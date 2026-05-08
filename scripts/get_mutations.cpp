// Extract cumulative nucleotide mutations per MAT node (same preorder as UShER depth_first_expansion)
// and write Parquet for Python (pyarrow): columns node_name, nt_mutations (;-separated).
//

#include "mutation_annotated_tree.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <arrow/api.h>
#include <arrow/array/builder_binary.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace MAT = Mutation_Annotated_Tree;

std::string haplotype_mutations(MAT::Node* tip) {
  std::vector<MAT::Mutation> mutations;
  while (tip != nullptr) {
    for (const MAT::Mutation& mut : tip->mutations) {
      if (mut.is_masked()) {
        continue;
      }
      bool site_already_present = false;
      for (const MAT::Mutation& existing : mutations) {
        if (existing.position == mut.position) {
          site_already_present = true;
          break;
        }
      }
      if (!site_already_present) {
        mutations.emplace_back(mut);
      }
    }
    tip = tip->parent;
  }
  std::string out;
  out.reserve(mutations.size() * 12);
  for (const MAT::Mutation& m : mutations) {
    if (!out.empty()) {
      out.push_back(';');
    }
    out += MAT::get_nuc(m.ref_nuc) + std::to_string(m.position) + MAT::get_nuc(m.mut_nuc);
  }
  return out;
}

void fill_haplotypes_parallel(
    const std::vector<MAT::Node*>& dfs_preorder,
    std::vector<std::string>& node_names,
    std::vector<std::string>& nt_mut_columns) {
  const std::size_t n = dfs_preorder.size();
  node_names.resize(n);
  nt_mut_columns.resize(n);

  tbb::parallel_for(
      tbb::blocked_range<std::size_t>(0, n),
      [&](const tbb::blocked_range<std::size_t>& r) {
        for (std::size_t i = r.begin(); i < r.end(); ++i) {
          MAT::Node* node = dfs_preorder[i];
          node_names[i] = node->identifier;
          nt_mut_columns[i] = haplotype_mutations(node);
        }
      });
}

void usage(const char* prog) {
  std::cerr << "Usage: " << prog << " --mat <tree.pb[.gz]> --output <out.parquet>\n";
}

arrow::Status write_parquet_two_columns(
    const std::vector<std::string>& node_names,
    const std::vector<std::string>& nt_mutations,
    const std::string& path) {
  arrow::LargeStringBuilder node_builder;
  arrow::LargeStringBuilder mut_builder;
  ARROW_RETURN_NOT_OK(node_builder.AppendValues(node_names));
  ARROW_RETURN_NOT_OK(mut_builder.AppendValues(nt_mutations));

  std::shared_ptr<arrow::Array> arr_nodes;
  std::shared_ptr<arrow::Array> arr_muts;
  ARROW_RETURN_NOT_OK(node_builder.Finish(&arr_nodes));
  ARROW_RETURN_NOT_OK(mut_builder.Finish(&arr_muts));

  auto schema = arrow::schema({
      arrow::field("node_name", arrow::large_utf8()),
      arrow::field("nt_mutations", arrow::large_utf8()),
  });

  auto table = arrow::Table::Make(schema, {arr_nodes, arr_muts});

  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(path));
  ARROW_RETURN_NOT_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out, 1024 * 1024));
  return arrow::Status::OK();
}

}  // namespace

int main(int argc, char** argv) {
  std::string mat_path;
  std::string out_path;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--mat" && i + 1 < argc) {
      mat_path = argv[++i];
    } else if (a == "--output" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    } else {
      std::cerr << "unknown argument: " << a << "\n";
      usage(argv[0]);
      return 1;
    }
  }

  if (mat_path.empty() || out_path.empty()) {
    usage(argv[0]);
    return 1;
  }

  try {
    Mutation_Annotated_Tree::Tree tree =
        Mutation_Annotated_Tree::load_mutation_annotated_tree(mat_path);
    if (tree.root == nullptr) {
      throw std::runtime_error("MAT has null root");
    }

    const std::vector<Mutation_Annotated_Tree::Node*> dfs =
        tree.depth_first_expansion();
    std::vector<std::string> names;
    std::vector<std::string> mcol;
    fill_haplotypes_parallel(dfs, names, mcol);

    arrow::Status st = write_parquet_two_columns(names, mcol, out_path);
    if (!st.ok()) {
      std::cerr << "parquet write failed: " << st.ToString() << '\n';
      return 1;
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "get_mutations: " << e.what() << '\n';
    return 1;
  }
}
