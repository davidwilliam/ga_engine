# Bibliography Notes

## BibTeX Setup

The paper now uses a proper BibTeX file (`references.bib`) instead of manual `\thebibliography` entries.

### Files
- **Bibliography database**: `references.bib`
- **LaTeX file**: `ga_crypto_ml_paper_compact.tex`

### Compilation
To compile the paper with bibliography, use:
```bash
pdflatex ga_crypto_ml_paper_compact.tex
bibtex ga_crypto_ml_paper_compact
pdflatex ga_crypto_ml_paper_compact.tex
pdflatex ga_crypto_ml_paper_compact.tex
```

Or if using Overleaf, it will handle this automatically.

## Verified References

All references have been verified against actual publications:

### ✅ Fully Verified

1. **vaikuntanathan2015** - Vinod Vaikuntanathan's lecture at Simons Institute
   - Source: Simons Institute Cryptography Boot Camp, 2015
   - URL: https://simons.berkeley.edu/talks/vinod-vaikuntanathan-2015-06-18

2. **hestenes1984** - Clifford Algebra to Geometric Calculus
   - Authors: David Hestenes, Garret Sobczyk
   - Publisher: Springer, 1984
   - DOI: 10.1007/978-94-009-6292-7
   - ISBN: 978-90-277-1673-6

3. **dorst2007** - Geometric Algebra for Computer Science
   - Authors: Leo Dorst, Daniel Fontijne, Stephen Mann
   - Publisher: Morgan Kaufmann, 2007
   - ISBN: 978-0-12-369465-2

4. **vince2008** - Geometric Algebra for Computer Graphics
   - Author: John Vince
   - Publisher: Springer, 2008
   - DOI: 10.1007/978-1-84628-997-2
   - ISBN: 978-1-84628-996-5

5. **regev2009** - On lattices, learning with errors, random linear codes, and cryptography
   - Author: Oded Regev
   - Journal: Journal of the ACM, Vol. 56(6), pp. 34:1--34:40
   - DOI: 10.1145/1568318.1568324
   - Year: 2009

6. **peikert2016** - A decade of lattice cryptography
   - Author: Chris Peikert
   - Journal: Foundations and Trends in Theoretical Computer Science, Vol. 10(4), pp. 283--424
   - DOI: 10.1561/0400000074
   - Year: 2016

7. **hoffstein1998** - NTRU: A ring-based public key cryptosystem
   - Authors: Jeffrey Hoffstein, Jill Pipher, Joseph H. Silverman
   - Conference: Algorithmic Number Theory (ANTS 1998)
   - Series: LNCS Vol. 1423, pp. 267--288
   - Publisher: Springer
   - DOI: 10.1007/BFb0054868
   - Year: 1998

8. **breuils2018** - Quadric Conformal Geometric Algebra
   - Authors: Stéphane Breuils, Vincent Nozick, Akihiro Sugimoto, Eckhard Hitzer
   - Journal: Advances in Applied Clifford Algebras, Vol. 28(2), Article 35
   - DOI: 10.1007/s00006-018-0851-1
   - Year: 2018

9. **fontijne2007** - Efficient Implementation of Geometric Algebra
   - Author: Daniel Fontijne
   - Type: PhD thesis, University of Amsterdam
   - URL: https://hdl.handle.net/11245/1.274934
   - Year: 2007

10. **clifford2019** - pygae/clifford: Geometric Algebra for Python
    - Authors: Hugo Hadfield, Eric Wieser, Alex Arsenovic, Robert Kern, The Pygae Team
    - Type: Software (Zenodo)
    - DOI: 10.5281/zenodo.1453978
    - URL: https://github.com/pygae/clifford
    - Version: 1.2.0 (2019)

11. **criterion2023** - Criterion.rs benchmarking library
    - Author: Brook Heisler
    - Type: Software library
    - URL: https://github.com/bheisler/criterion.rs

### ⚠️ Needs Verification

12. **dorstprofile** - Personal communication with Leo Dorst
    - Type: Personal communication at ICGA 2022, Denver
    - **Note**: This is accurate per the conversation history, but journals may have specific requirements for citing personal communications

13. **josipovic2024** - Hardware acceleration for lattice-based cryptography
    - **STATUS**: This is a PLACEHOLDER reference
    - The paper cites hardware acceleration work showing 1.54-3.07× NTRU speedup on Apple M1/M3
    - **ACTION REQUIRED**: Find and verify the actual publication
    - Possible sources to search:
      - IEEE Xplore (IEEE Transactions on Computers, ISCA, MICRO, etc.)
      - ACM Digital Library (embedded systems conferences)
      - arXiv.org (preprints)
    - Search terms: "NTRU hardware acceleration M1 M3 Apple Silicon"

## Citation Keys Used in Paper

The following citation keys are used in the LaTeX file:

- `\cite{vaikuntanathan2015}` - Line 51 (Introduction)
- `\cite{dorstprofile}` - Line 53 (Introduction)
- `\cite{josipovic2024}` - Lines 64, 135, 260 (multiple locations)
- `\cite{hestenes1984,dorst2007}` - Line 91 (Background)
- `\cite{vince2008}` - Line 111 (Background)
- `\cite{regev2009,peikert2016}` - Line 115 (Background)
- `\cite{hoffstein1998}` - Line 115 (Background)
- `\cite{breuils2018}` - Line 132 (Related Work)
- `\cite{fontijne2007}` - Line 133 (Related Work)
- `\cite{clifford2019}` - Line 134 (Related Work)
- `\cite{criterion2023}` - Line 214 (Methodology)

## Important Notes

### Reference Style
The Royal Society template uses the `plain` bibliography style. You may need to adjust to match the journal's specific requirements (e.g., `rsc` style if available).

### DOIs and URLs
All major references include DOIs where available. URLs are provided for:
- Online lectures (Vaikuntanathan)
- Theses (Fontijne)
- Software (Clifford, Criterion.rs)

### Software Citations
The paper cites two software packages:
1. **clifford2019**: Python library for geometric algebra (properly cited via Zenodo DOI)
2. **criterion2023**: Rust benchmarking framework (GitHub URL)

Both are legitimate citations for academic papers discussing computational methods.

## Next Steps

1. **Find josipovic2024 reference** - This is the most critical missing piece
   - Search IEEE Xplore and ACM Digital Library
   - Check recent ISCA, MICRO, DAC, or embedded systems conferences
   - If not found, consider removing the comparison or noting it as "unpublished results"

2. **Verify personal communication** - Check journal guidelines for citing personal communications
   - Some journals require written permission
   - Others prefer footnotes rather than bibliography entries

3. **Check bibliography style** - Confirm `plain` is acceptable or if Royal Society has a specific .bst file

4. **Compile and verify** - Run BibTeX compilation to ensure all references resolve correctly

## Reference Format Examples

The BibTeX file uses standard entry types:
- `@book` - Books (Hestenes, Dorst, Vince)
- `@article` - Journal articles (Regev, Peikert, Breuils)
- `@inproceedings` - Conference papers (Hoffstein, josipovic2024)
- `@phdthesis` - PhD dissertations (Fontijne)
- `@software` - Software packages (Clifford)
- `@misc` - Other types (Vaikuntanathan, Criterion.rs)

All entries include standard fields: author, title, year, and publication details (journal, publisher, DOI, etc.).
