mkdir ts_package
cd ts_package

# --- Clone and checkout matching versions for tree-sitter 0.20.2 ---

# C parser (compatible with 0.20.x)
git clone https://github.com/tree-sitter/tree-sitter-c.git
cd tree-sitter-c
git checkout v0.20.1
cd ..

# C++ parser (compatible with 0.20.x)
git clone https://github.com/tree-sitter/tree-sitter-cpp.git
cd tree-sitter-cpp
git checkout v0.20.0
cd ..

# Java parser (compatible with 0.20.x)
git clone https://github.com/tree-sitter/tree-sitter-java.git
cd tree-sitter-java
git checkout v0.20.0
cd ..

cd ..

# --- Build the tree-sitter shared libraries ---
python build_ts_lib.py