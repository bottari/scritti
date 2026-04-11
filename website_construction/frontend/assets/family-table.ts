type Person = {
  id: string;
  name: string;
  birthYear?: number;
  deathYear?: number;
  branch?: string;
};

const tableBody = document.getElementById("family-table-body");
const searchInput = document.getElementById("family-search") as HTMLInputElement | null;
const branchFilter = document.getElementById("branch-filter") as HTMLSelectElement | null;

let allPeople: Person[] = [];

async function loadFamilyTable(): Promise<void> {
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

function populateBranchFilter(people: Person[]): void {
  if (!branchFilter) {
    return;
  }

  const branches = Array.from(
    new Set(people.map((person) => person.branch || "other"))
  ).sort();

  branches.forEach((branch) => {
    const option = document.createElement("option");
    option.value = branch;
    option.textContent = capitalize(branch);
    branchFilter.appendChild(option);
  });
}

function renderRows(people: Person[]): void {
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

function applyFilters(): void {
  const query = searchInput?.value.toLowerCase().trim() || "";
  const branch = branchFilter?.value || "all";

  const filtered = allPeople.filter((person) => {
    const matchesName = person.name.toLowerCase().includes(query);
    const matchesBranch = branch === "all" || (person.branch || "other") === branch;
    return matchesName && matchesBranch;
  });

  renderRows(filtered);
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

searchInput?.addEventListener("input", applyFilters);
branchFilter?.addEventListener("change", applyFilters);

loadFamilyTable();
