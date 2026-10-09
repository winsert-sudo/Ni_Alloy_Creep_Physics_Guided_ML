# Data access

Project NIMS access was through an officially registered and approved user account, as confirmed in the C4 task instructions. This author-provided acquisition fact does not establish redistribution permission or complete extraction history. NIMS experimental and processed row data are excluded.

Eligible users should consult https://metallicmaterials.nims.go.jp/, https://cds.nims.go.jp/ and https://fds.nims.go.jp/ for registration and service approval, and comply with the applicable terms: https://mits.nims.go.jp/agreement/MatNavi_agreement_en.pdf (Japanese original controls). No credentials or automated acquisition routines are supplied.

ORNL-5479 is publicly accessible at https://www.osti.gov/servlets/purl/6402711; M. K. Booker (1979), DOI 10.2172/6402711. Public access and an unlimited-distribution marking do not by themselves establish the project's extracted-data sublicense. C3 verified no applicable row-data redistribution grant; ORNL rows and a numerical demonstration are therefore excluded. See https://www.osti.gov/disclaim.

[Source registry](metadata/SOURCE_REGISTRY_PUBLIC.csv) supplies 44 metadata-only locators: 39 NIMS source groups and five groups sharing one ORNL report. NIMS individual sheet titles/authors/years/DOIs remain unverified. No experimental reference codes, heat IDs or source-to-state/record mapping are included. Exact NIMS source matching is 38 exact and one datasheet-only; missing material condition/reference information must not be invented.

User-supplied lawful data must satisfy [schema](docs/DATA_SCHEMA.md) and the script-specific [input contract](src/CODE_INPUTS.json). Fields include condition units K/MPa, chemistry metadata, source grouping and log10 life in hours. These are field definitions, not provided values. Preserve missingness semantics, source-disjoint folds and training-fold-only transformations. A presence check does not establish schema compatibility or rights. Exact benchmark access/reconstruction, private fold alignment and ancillary audits are not independently demonstrated.

Project-created repository content is licensed under BSD-3-Clause; see [license scope](LICENSE_SCOPE.md). This license does not cover NIMS/ORNL row-level data or referenced source works. Data-access and exact-reproduction limitations remain in force.
