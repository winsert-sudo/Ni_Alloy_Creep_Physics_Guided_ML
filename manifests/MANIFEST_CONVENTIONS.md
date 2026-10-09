# Inventory convention

PUBLIC_FILE_MANIFEST.csv inventories every staged file, including itself and CHECKSUMS_SHA256.txt. Recursive hashes for those two inventory carriers are deliberately blank in the CSV. CHECKSUMS_SHA256.txt hashes every staged file except itself, including the completed CSV. From the root run `python -B tests/check_package.py` or `sha256sum --check manifests/CHECKSUMS_SHA256.txt`. Neither command trains a model. Internal traceability, comparisons and author approval reports are stored outside this candidate.
