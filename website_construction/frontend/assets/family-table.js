const tableBody = document.getElementById("family-table-body");
const searchInput = document.getElementById("family-search");
const branchFilter = document.getElementById("branch-filter");
let allPeople = [];
async function loadFamilyTable() {
  if (!tableBody) {
    return;
  }
  try {
    const response = await fetch("/api/family");
    allPeople = await response.json();
    populateBranchFilter(allPeople);
    renderRows(allPeople);
  } catch (error) {
    tableBody.innerHTML = `<tr><td colspan="4">Unable to load family data right now.</td></tr>`;
  }
}
function populateBranchFilter(people) {
  if (!branchFilter) {
    return;
  }
  const branches = Array.from(new Set(people.map((person) => person.branch || "other"))).sort();
  branches.forEach((branch) => {
    const option = document.createElement("option");
    option.value = branch;
    option.textContent = capitalize(branch);
    branchFilter.appendChild(option);
  });
}
function renderRows(people) {
  if (!tableBody) {
    return;
  }
  if (!people.length) {
    tableBody.innerHTML = `<tr><td colspan="4">No family members match the current filters.</td></tr>`;
    return;
  }
  tableBody.innerHTML = people
    .map(
      (person) => `
        <tr data-person-id="${person.id}">
          <td>${person.name}</td>
          <td>${person.birthYear ?? "—"}</td>
          <td>${person.deathYear ?? "—"}</td>
          <td>${capitalize(person.branch || "other")}</td>
        </tr>
      `
    )
    .join("");
  tableBody.querySelectorAll("tr").forEach((row) => {
    row.addEventListener("click", () => {
      const id = row.getAttribute("data-person-id");
      if (id) {
        window.location.href = `/person?id=${encodeURIComponent(id)}`;
      }
    });
  });
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
function capitalize(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
searchInput?.addEventListener("input", applyFilters);
branchFilter?.addEventListener("change", applyFilters);
loadFamilyTable();
