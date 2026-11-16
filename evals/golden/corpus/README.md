# Golden-set corpus

Plain-text contracts from the **Contract Understanding Atticus Dataset (CUAD)
v1**, copied byte-for-byte from `full_contract_txt/` for the contracts the
golden case set (`../cases.json`) uses. Vendored so the CI eval gate can run
without the full dataset. Regenerate with `scripts/vendor_golden_corpus.py`.

> Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021). *CUAD: An
> Expert-Annotated NLP Dataset for Legal Contract Review.* NeurIPS Datasets and
> Benchmarks Track.

Distributed by The Atticus Project (https://www.atticusprojectai.org/cuad)
under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Unmodified.
The Atticus Project makes no representations about the licence status of the
underlying contracts, which are public SEC EDGAR filings.
