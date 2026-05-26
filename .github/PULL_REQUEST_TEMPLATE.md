# Pull Request Template

## 📋 Description

**Closes**: # (issue number)

**Type of change**: *(check one)*

- [ ] 🚀 Feature (new functionality)
- [ ] 🐛 Bug fix (problem resolution)
- [ ] 📝 Documentation (wiki, README, comments)
- [ ] 🎨 Style (formatting, no code change)
- [ ] ♻️ Refactor (code improvement, no functional change)
- [ ] ✅ Test (adding/fixing tests)
- [ ] 🔧 Chore (maintenance, dependencies)
- [ ] 🎬 Content (AI-generated images, videos, PPT)

**Summary**:
> Briefly describe what this PR does and why.

---

## 🔍 Changes Made

### Added
- [ ] New feature 1
- [ ] New feature 2

### Changed
- [ ] Modified component 1
- [ ] Modified component 2

### Fixed
- [ ] Bug fix 1
- [ ] Bug fix 2

### Removed
- [ ] Deprecated component

---

## 🧪 Testing Performed

- [ ] Unit tests pass (`pytest tests/`)
- [ ] Manual testing completed
- [ ] API tested with curl/Postman
- [ ] Frontend tested in browser
- [ ] Docker build successful *(if applicable)*

### Test Details

```bash
# Example test commands run
curl http://localhost:8000/health
python scripts/test_generation.py
```

---

## 📸 Screenshots / Screen Recordings

| Before | After |
|---|---|
| *(screenshot)* | *(screenshot)* |

*(Delete if not applicable)*

---

## ✅ Checklist

### Code Quality
- [ ] Code follows project style guide (PEP 8)
- [ ] Type hints added for function arguments
- [ ] No commented-out code
- [ ] No debugging `print` statements left
- [ ] Variable names are meaningful

### Documentation
- [ ] README updated if needed
- [ ] Wiki updated if needed
- [ ] API documentation updated *(if endpoints changed)*
- [ ] Docstrings added for new functions

### Git Best Practices
- [ ] Branch name follows convention (`feature/xxx`, `bugfix/xxx`)
- [ ] Commit messages are clear and descriptive
- [ ] No merge conflicts with `main`
- [ ] Commits are atomic (one logical change per commit)

### Testing
- [ ] New code has tests
- [ ] Existing tests still pass
- [ ] Edge cases handled

### Security & Performance
- [ ] No sensitive data (API keys, passwords) exposed
- [ ] Rate limiting respected
- [ ] Performance impact considered

---

## 📊 Impact Assessment

| Aspect | Impact Level | Notes |
|---|---|---|
| Backward Compatibility | High / Medium / Low / None | |
| API Changes | Yes / No | |
| Database Changes | Yes / No | |
| Environment Variables | Yes / No | |

---

## 🔗 Related Issues / PRs

- Related to # (issue number)
- Depends on # (PR number)
- Blocks # (issue number)

---

## 📝 Additional Notes

> Any additional context, decisions made, or questions for reviewers?

---

## 👀 Reviewer Instructions

**Focus areas:**
1.
2.
3.

**To test this PR:**

```bash
git checkout main
git pull origin main
git checkout -b test/PR-XXX
git pull origin pull/XXX/head
# Run tests and verify
```

---

## ✅ Definition of Done

- [ ] Code reviewed *(even if self-review)*
- [ ] All tests pass
- [ ] Documentation updated
- [ ] No blocking issues
- [ ] Branch merged
- [ ] Branch deleted (local and remote)

---

<div align="center">

Thank you for your contribution! 🚀

**#ToTheNEXTLevel**

</div>
