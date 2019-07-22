# Artificial Neural Networks C library
TBD
# libfixmath

```bash
git clone git@github.com:Ahmed-Zamouche/ann.git

git submodule init
git submodule update

cd libfixmath
git checkout -b ann
git apply ../libfixmath.patch
cd ..
```

# Build
```bash
make
```

# Run test
```bash
make test && ./build/bin/TEST_ann
```