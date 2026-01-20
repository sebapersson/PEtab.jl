# [Contributing to PEtab.jl](@id contribute)

First, thanks for taking the time to contribute to PEtab.jl! Contributions of all kinds
helps make the package better for everyone.

## Ways to contribute

Contributions are welcome in the form of:

- Reporting bugs
- Suggesting features or enhancements
- Submitting fixes or improvements via pull requests (PRs)
- Improving documentation (typos, examples, clarifications, tutorials)

## Reporting bugs

If you find a bug, please open an issue on
[GitHub](https://github.com/sebapersson/PEtab.jl/issues) and include:

- A clear, descriptive title
- Steps to reproduce the problem, ideally with a minimal working example (MRE). For more
  about MREs, see
  [Wikipedia](https://en.wikipedia.org/wiki/Minimal_reproducible_example) and
  [Stack Overflow](https://stackoverflow.com/help/minimal-reproducible-example).
- What you expected to happen
- What actually happened
- Relevant error messages, logs, or screenshots (if applicable)
- Version information (Julia version, PEtab.jl version, and OS)

## Suggesting enhancements

New ideas are welcome. Before opening a new issue, check the
[issue tracker](https://github.com/sebapersson/PEtab.jl/issues) to see whether it has
already been suggested. If not, open a new issue describing:

- The enhancement you would like
- Why it would be useful
- Any implementation ideas (optional, but helpful)

## Pull request process

Before submitting a PR:

1. Ensure there is an open issue related to the change (or create one).
2. Fork the repository and create a new branch from `main`:

   ```bash
   git checkout -b feature/my-feature
   ```

3. Make changes:
   - Keep changes focused and well-documented
   - Add or update tests where appropriate
   - If needed, update documentation
4. Run the test suite. From the repository root, launch Julia with `julia --project=.` and
   then run:

   ```julia
    ] test
   ```

5. Submit a pull request and reference the related issue (if applicable).

### Pull request checklist

- [ ] Tests added/updated as appropriate
- [ ] Documentation updated as needed
- [ ] Linked to an issue (if applicable)

## Code style and documentation guidelines

- Follow the formatting/style conventions enforced by Runic.jl.
- Write clear docstrings for exported functions (see the API documentation for examples).
- Prefer focused PRs that are easy to review.
