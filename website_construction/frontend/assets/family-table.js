const TOKEN_STORAGE_KEY = "family-edit-token";
const tableBody = document.getElementById("family-table-body");
const searchInput = document.getElementById("family-search");
const branchFilter = document.getElementById("branch-filter");
const tokenInput = document.getElementById("family-edit-token");
const addButton = document.getElementById("family-add-button");
const feedbackRoot = document.getElementById("family-feedback");
const editorRoot = document.getElementById("family-editor");
let allPeople = [];
let editingPersonId = null;
let isCreatingPerson = false;
async function loadFamilyTable() {
  if (!tableBody) {
    return;
  }
  const selectedBranch = branchFilter?.value || "all";
  const currentQuery = searchInput?.value || "";
  try {
    const response = await fetch("/api/family");
    if (!response.ok) {
      throw new Error(await extractErrorMessage(response));
    }
    allPeople = await response.json();
    populateBranchFilter(allPeople, selectedBranch);
    if (searchInput) {
      searchInput.value = currentQuery;
    }
    applyFilters();
    refreshOpenEditor();
  } catch (error) {
    tableBody.innerHTML = `<tr><td colspan="5">Unable to load family data right now.</td></tr>`;
    setFeedback(getErrorMessage(error), "error");
  }
}
function populateBranchFilter(people, selectedBranch) {
  if (!branchFilter) {
    return;
  }
  const branches = Array.from(new Set(people.map((person) => person.branch || "other"))).sort((left, right) =>
    left.localeCompare(right)
  );
  branchFilter.innerHTML = `<option value="all">All branches</option>`;
  branches.forEach((branch) => {
    const option = document.createElement("option");
    option.value = branch;
    option.textContent = capitalize(branch);
    branchFilter.appendChild(option);
  });
  branchFilter.value = branches.includes(selectedBranch) ? selectedBranch : "all";
}
function renderRows(people) {
  if (!tableBody) {
    return;
  }
  if (!people.length) {
    tableBody.innerHTML = `<tr><td colspan="5">No family members match the current filters.</td></tr>`;
    return;
  }
  tableBody.innerHTML = people
    .map(
      (person) => `
        <tr data-person-id="${escapeHtml(person.id)}">
          <td>
            <a class="family-name-link" href="/person?id=${encodeURIComponent(person.id)}">
              ${escapeHtml(person.name)}
            </a>
          </td>
          <td>${formatYear(person.birthYear)}</td>
          <td>${formatYear(person.deathYear)}</td>
          <td>${escapeHtml(capitalize(person.branch || "other"))}</td>
          <td>
            <div class="family-row-actions">
              <button
                type="button"
                class="family-row-button"
                data-family-edit="${escapeHtml(person.id)}"
              >
                Edit
              </button>
            </div>
          </td>
        </tr>
      `
    )
    .join("");
}
function applyFilters() {
  const query = (searchInput?.value || "").toLowerCase().trim();
  const branch = branchFilter?.value || "all";
  const filtered = allPeople.filter((person) => {
    const matchesName = person.name.toLowerCase().includes(query);
    const matchesBranch = branch === "all" || (person.branch || "other") === branch;
    return matchesName && matchesBranch;
  });
  renderRows(filtered);
}
function openCreateEditor() {
  isCreatingPerson = true;
  editingPersonId = null;
  clearFeedback();
  renderEditor(createBlankPerson(), "create");
}
function openEditEditor(personId) {
  const person = findPerson(personId);
  if (!person) {
    setFeedback("That family member is no longer available to edit.", "error");
    return;
  }
  isCreatingPerson = false;
  editingPersonId = personId;
  clearFeedback();
  renderEditor(person, "edit");
}
function closeEditor() {
  isCreatingPerson = false;
  editingPersonId = null;
  if (editorRoot) {
    editorRoot.innerHTML = "";
    editorRoot.classList.add("is-hidden");
  }
}
function refreshOpenEditor() {
  if (isCreatingPerson) {
    renderEditor(createBlankPerson(), "create");
    return;
  }
  if (!editingPersonId) {
    return;
  }
  const person = findPerson(editingPersonId);
  if (!person) {
    closeEditor();
    return;
  }
  renderEditor(person, "edit");
}
function renderEditor(person, mode) {
  if (!editorRoot) {
    return;
  }
  const linkedSpouseId = resolvePersonIdByName(person.spouse);
  const isEditMode = mode === "edit";
  editorRoot.classList.remove("is-hidden");
  editorRoot.innerHTML = `
    <div class="family-editor-header">
      <div>
        <h3>${isEditMode ? "Edit family member" : "Add family member"}</h3>
        <p class="family-helper-text">
          ${
            isEditMode
              ? "Update the record, then save to persist the change to the server-side family data file."
              : "Create a new family record. The id becomes the stable link key for relationships."
          }
        </p>
      </div>
      <button type="button" class="tree-action-button" data-family-cancel>Cancel</button>
    </div>
    <form class="family-editor-form" data-editor-mode="${mode}">
      <div class="family-editor-grid">
        <div class="family-field">
          <label for="family-editor-id">Identifier</label>
          <input
            id="family-editor-id"
            name="id"
            value="${escapeHtml(person.id)}"
            ${isEditMode ? "readonly" : ""}
            data-auto-id="${isEditMode ? "false" : "true"}"
            required
          />
          <p class="family-helper-text">
            ${
              isEditMode
                ? "Existing records keep their id fixed so parent and child links stay stable."
                : "Use lowercase letters, numbers, and hyphens only."
            }
          </p>
        </div>
        <div class="family-field">
          <label for="family-editor-name">Name</label>
          <input id="family-editor-name" name="name" value="${escapeHtml(person.name)}" required />
        </div>
        <div class="family-field">
          <label for="family-editor-first-name">First Name</label>
          <input
            id="family-editor-first-name"
            name="firstName"
            value="${escapeHtml(readText(person.firstName))}"
          />
        </div>
        <div class="family-field">
          <label for="family-editor-last-name">Last Name</label>
          <input
            id="family-editor-last-name"
            name="lastName"
            value="${escapeHtml(readText(person.lastName))}"
          />
        </div>
        <div class="family-field">
          <label for="family-editor-branch">Relation / Branch</label>
          <input
            id="family-editor-branch"
            name="branch"
            value="${escapeHtml(readText(person.branch || "other"))}"
          />
        </div>
        <div class="family-field">
          <label for="family-editor-relation">Relation Label</label>
          <input
            id="family-editor-relation"
            name="relation"
            value="${escapeHtml(readText(person.relation))}"
          />
        </div>
        <div class="family-field">
          <label for="family-editor-birth-year">Birth Year</label>
          <input
            id="family-editor-birth-year"
            name="birthYear"
            type="number"
            value="${escapeHtml(formatInputYear(person.birthYear))}"
          />
        </div>
        <div class="family-field">
          <label for="family-editor-death-year">Death Year</label>
          <input
            id="family-editor-death-year"
            name="deathYear"
            type="number"
            value="${escapeHtml(formatInputYear(person.deathYear))}"
          />
        </div>
        <div class="family-field">
          <label for="family-editor-birth-date">Birth Date</label>
          <input
            id="family-editor-birth-date"
            name="birthDate"
            value="${escapeHtml(readText(person.birthDate))}"
          />
        </div>
        <div class="family-field">
          <label for="family-editor-death-date">Death Date</label>
          <input
            id="family-editor-death-date"
            name="deathDate"
            value="${escapeHtml(readText(person.deathDate))}"
          />
        </div>
        <div class="family-field">
          <label for="family-editor-birth-location">Birth Location</label>
          <input
            id="family-editor-birth-location"
            name="birthLocation"
            value="${escapeHtml(readText(person.birthLocation))}"
          />
        </div>
        <div class="family-field">
          <label for="family-editor-photo-path">Photo Path</label>
          <input
            id="family-editor-photo-path"
            name="photoPath"
            value="${escapeHtml(readText(person.photoPath))}"
          />
        </div>
        <div class="family-field">
          <label for="family-editor-spouse">Spouse / Partner</label>
          <input
            id="family-editor-spouse"
            name="spouse"
            value="${escapeHtml(readText(person.spouse))}"
          />
        </div>
        <div class="family-field">
          <label for="family-editor-linked-spouse">Link to Existing Record</label>
          <select id="family-editor-linked-spouse" name="linkedSpouseId">
            <option value="">No linked spouse record</option>
            ${renderPersonOptions(linkedSpouseId ? [linkedSpouseId] : [], person.id)}
          </select>
        </div>
        <div class="family-field">
          <label for="family-editor-marriage-date">Marriage Date</label>
          <input
            id="family-editor-marriage-date"
            name="marriageDate"
            value="${escapeHtml(readText(person.marriageDate))}"
          />
        </div>
        <div class="family-field">
          <label for="family-editor-father">Father</label>
          <input
            id="family-editor-father"
            name="father"
            value="${escapeHtml(readText(person.father))}"
          />
        </div>
        <div class="family-field">
          <label for="family-editor-mother">Mother</label>
          <input
            id="family-editor-mother"
            name="mother"
            value="${escapeHtml(readText(person.mother))}"
          />
        </div>
        <div class="family-field is-wide">
          <label for="family-editor-parents">Parents</label>
          <select id="family-editor-parents" name="parents" multiple>
            ${renderPersonOptions(person.parents || [], person.id)}
          </select>
          <p class="family-helper-text">
            Linked parent ids drive the tree. When parent ids are set, the saved father and mother names are refreshed from those linked records.
          </p>
        </div>
        <div class="family-field is-wide">
          <label for="family-editor-children">Children</label>
          <select id="family-editor-children" name="children" multiple>
            ${renderPersonOptions(person.children || [], person.id)}
          </select>
        </div>
        <div class="family-field is-wide">
          <label for="family-editor-bio">Biography</label>
          <textarea id="family-editor-bio" name="bio">${escapeHtml(readText(person.bio))}</textarea>
        </div>
      </div>
      <div class="family-form-actions">
        <button type="submit" class="button">${isEditMode ? "Save changes" : "Create member"}</button>
        <button type="button" class="button button-secondary" data-family-cancel>Cancel</button>
      </div>
    </form>
  `;
  editorRoot.scrollIntoView({ block: "start", behavior: "smooth" });
}
function renderPersonOptions(selectedIds, currentPersonId) {
  const selectedSet = new Set(selectedIds);
  const renderedOptions = allPeople
    .filter((candidate) => candidate.id !== currentPersonId)
    .map((candidate) => {
      const selected = selectedSet.has(candidate.id) ? "selected" : "";
      const branch = candidate.branch ? ` (${capitalize(candidate.branch)})` : "";
      return `
        <option value="${escapeHtml(candidate.id)}" ${selected}>
          ${escapeHtml(candidate.name)}${escapeHtml(branch)}
        </option>
      `;
    })
    .join("");
  const unresolvedSelections = selectedIds
    .filter(
      (selectedId) => selectedId !== currentPersonId && !allPeople.some((candidate) => candidate.id === selectedId)
    )
    .map(
      (selectedId) => `
        <option value="${escapeHtml(selectedId)}" selected>
          ${escapeHtml(selectedId)} (unlinked)
        </option>
      `
    )
    .join("");
  return `${unresolvedSelections}${renderedOptions}`;
}
async function handleEditorSubmit(event) {
  const form = event.target;
  if (!form || !form.matches(".family-editor-form")) {
    return;
  }
  event.preventDefault();
  const mode = form.dataset.editorMode === "edit" ? "edit" : "create";
  const personId = readFormText(form, "id");
  const existingRecord = mode === "edit" ? findPerson(personId) : undefined;
  if (mode === "edit" && !existingRecord) {
    setFeedback("This record could not be found anymore. Reload the page and try again.", "error");
    return;
  }
  const token = tokenInput?.value.trim() || readStoredToken();
  if (!token) {
    setFeedback("Enter the family edit token before saving changes.", "error");
    tokenInput?.focus();
    return;
  }
  const payload = buildPayloadFromForm(form, existingRecord);
  const url = mode === "edit" ? `/api/family/${encodeURIComponent(personId)}` : "/api/family";
  const method = mode === "edit" ? "PUT" : "POST";
  try {
    const response = await fetch(url, {
      method,
      headers: {
        "Content-Type": "application/json",
        "X-Family-Edit-Token": token,
      },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error(await extractErrorMessage(response));
    }
    writeStoredToken(token);
    closeEditor();
    setFeedback(mode === "edit" ? `Saved changes for ${payload.name}.` : `Created ${payload.name}.`, "success");
    await loadFamilyTable();
    window.dispatchEvent(new CustomEvent("family-data-changed"));
  } catch (error) {
    setFeedback(getErrorMessage(error), "error");
  }
}
function buildPayloadFromForm(form, existingRecord) {
  const base = existingRecord ? clonePerson(existingRecord) : createBlankPerson();
  const linkedSpouseId = readFormText(form, "linkedSpouseId");
  const linkedSpouse = linkedSpouseId ? findPerson(linkedSpouseId) : undefined;
  base.id = readFormText(form, "id");
  base.name = readFormText(form, "name");
  base.firstName = readFormText(form, "firstName");
  base.lastName = readFormText(form, "lastName");
  base.birthDate = readFormText(form, "birthDate");
  base.birthLocation = readFormText(form, "birthLocation");
  base.birthYear = readOptionalNumber(form, "birthYear");
  base.deathDate = readFormText(form, "deathDate");
  base.deathYear = readOptionalNumber(form, "deathYear");
  base.branch = readFormText(form, "branch") || "other";
  base.relation = readFormText(form, "relation");
  base.spouse = linkedSpouse?.name || readFormText(form, "spouse");
  base.marriageDate = readFormText(form, "marriageDate");
  base.father = readFormText(form, "father");
  base.mother = readFormText(form, "mother");
  base.parents = readMultiSelect(form, "parents");
  base.children = readMultiSelect(form, "children");
  base.photoPath = readFormText(form, "photoPath");
  base.bio = readFormText(form, "bio");
  return base;
}
function handleTableInteraction(event) {
  const target = event.target;
  if (!target) {
    return;
  }
  const editButton = target.closest("[data-family-edit]");
  if (!editButton) {
    return;
  }
  const personId = editButton.dataset.familyEdit;
  if (personId) {
    openEditEditor(personId);
  }
}
function handleEditorClick(event) {
  const target = event.target;
  if (!target) {
    return;
  }
  const cancelButton = target.closest("[data-family-cancel]");
  if (cancelButton) {
    closeEditor();
  }
}
function handleEditorInput(event) {
  const target = event.target;
  if (!target || !editorRoot) {
    return;
  }
  const form = target.closest(".family-editor-form");
  if (!form) {
    return;
  }
  const idInput = form.querySelector('input[name="id"]');
  const nameInput = form.querySelector('input[name="name"]');
  if (!idInput || !nameInput) {
    return;
  }
  if (target === idInput && form.dataset.editorMode === "create") {
    idInput.dataset.autoId = "false";
  }
  if (target === nameInput && form.dataset.editorMode === "create" && idInput.dataset.autoId === "true") {
    idInput.value = slugify(nameInput.value);
  }
}
function handleEditorChange(event) {
  const target = event.target;
  if (!target) {
    return;
  }
  if (target.name === "linkedSpouseId") {
    const form = target.closest(".family-editor-form");
    const spouseInput = form?.querySelector('input[name="spouse"]');
    const linkedSpouse = findPerson(target.value);
    if (spouseInput && linkedSpouse) {
      spouseInput.value = linkedSpouse.name;
    }
  }
}
function createBlankPerson() {
  return {
    id: "",
    name: "",
    firstName: "",
    lastName: "",
    birthDate: "",
    birthLocation: "",
    birthYear: null,
    deathDate: "",
    deathYear: null,
    spouse: "",
    marriageDate: "",
    father: "",
    mother: "",
    bio: "",
    relation: "",
    photoPath: "",
    parents: [],
    children: [],
    branch: "other",
  };
}
function clonePerson(person) {
  return {
    ...person,
    parents: [...(person.parents || [])],
    children: [...(person.children || [])],
  };
}
function findPerson(personId) {
  return allPeople.find((person) => person.id === personId);
}
function resolvePersonIdByName(name) {
  const normalized = normalizeName(name);
  if (!normalized) {
    return undefined;
  }
  return allPeople.find((person) => normalizeName(person.name) === normalized)?.id;
}
function readFormText(form, name) {
  const field = form.elements.namedItem(name);
  if (!(field instanceof HTMLInputElement || field instanceof HTMLTextAreaElement || field instanceof HTMLSelectElement)) {
    return "";
  }
  return field.value.trim();
}
function readOptionalNumber(form, name) {
  const rawValue = readFormText(form, name);
  if (!rawValue) {
    return null;
  }
  const parsed = Number(rawValue);
  return Number.isFinite(parsed) ? parsed : null;
}
function readMultiSelect(form, name) {
  const field = form.elements.namedItem(name);
  if (!(field instanceof HTMLSelectElement)) {
    return [];
  }
  return Array.from(field.selectedOptions)
    .map((option) => option.value)
    .filter((value) => value.length > 0);
}
function setFeedback(message, tone) {
  if (!feedbackRoot) {
    return;
  }
  feedbackRoot.textContent = message;
  feedbackRoot.classList.remove("is-hidden", "is-success", "is-error");
  feedbackRoot.classList.add(tone === "success" ? "is-success" : "is-error");
}
function clearFeedback() {
  if (!feedbackRoot) {
    return;
  }
  feedbackRoot.textContent = "";
  feedbackRoot.classList.add("is-hidden");
  feedbackRoot.classList.remove("is-success", "is-error");
}
function writeStoredToken(token) {
  try {
    window.localStorage.setItem(TOKEN_STORAGE_KEY, token);
  } catch (error) {
    // Ignore storage failures and keep the current page usable.
  }
}
function readStoredToken() {
  try {
    return window.localStorage.getItem(TOKEN_STORAGE_KEY) || "";
  } catch (error) {
    return "";
  }
}
function restoreStoredToken() {
  if (tokenInput && !tokenInput.value) {
    tokenInput.value = readStoredToken();
  }
}
async function extractErrorMessage(response) {
  try {
    const payload = await response.json();
    return payload.detail || `Request failed with status ${response.status}.`;
  } catch (error) {
    return `Request failed with status ${response.status}.`;
  }
}
function getErrorMessage(error) {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return "Something went wrong while saving the family record.";
}
function formatYear(value) {
  return value === null || value === undefined ? "&mdash;" : escapeHtml(String(value));
}
function formatInputYear(value) {
  return value === null || value === undefined ? "" : String(value);
}
function readText(value) {
  return typeof value === "string" ? value : "";
}
function normalizeName(value) {
  return (value || "").trim().toLowerCase();
}
function slugify(value) {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}
function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
function capitalize(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
searchInput?.addEventListener("input", applyFilters);
branchFilter?.addEventListener("change", applyFilters);
if (tokenInput) {
  tokenInput.addEventListener("input", () => writeStoredToken(tokenInput.value.trim()));
}
tableBody?.addEventListener("click", handleTableInteraction);
addButton?.addEventListener("click", openCreateEditor);
editorRoot?.addEventListener("submit", (event) => {
  void handleEditorSubmit(event);
});
editorRoot?.addEventListener("click", handleEditorClick);
editorRoot?.addEventListener("input", handleEditorInput);
editorRoot?.addEventListener("change", handleEditorChange);
restoreStoredToken();
void loadFamilyTable();
