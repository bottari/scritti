type PersonSummary = {
  id: string;
  name: string;
  birthYear?: number;
  deathYear?: number;
  spouse?: string;
  parents?: string[];
  children?: string[];
  branch?: string;
};

type DescendancyNode = {
  id: string;
  person: PersonSummary;
  spouse?: PersonSummary;
  spouseName?: string;
  children: DescendancyNode[];
};

const treeRoot = document.getElementById("family-tree-root");
const peopleById = new Map<string, PersonSummary>();
const peopleByName = new Map<string, PersonSummary>();

let visibleRootIds: string[] = [];
let selectedRootId: string | null = null;
let collapsedBranchIds = new Set<string>();

treeRoot?.addEventListener("click", handleTreeInteraction);

async function loadFamilyTree(): Promise<void> {
  if (!treeRoot) {
    return;
  }

  treeRoot.innerHTML = `<p class="message">Loading family tree...</p>`;

  try {
    const response = await fetch("/api/family");
    const people = (await response.json()) as PersonSummary[];

    populatePeopleMaps(people);

    const visibleRoots = buildVisibleRoots(people);
    if (!visibleRoots.length) {
      treeRoot.innerHTML = `<p class="message">No tree data is available yet.</p>`;
      return;
    }

    visibleRootIds = visibleRoots.map((person) => person.id);
    if (!selectedRootId || !visibleRootIds.includes(selectedRootId)) {
      selectedRootId = visibleRootIds[0];
    }

    renderTreeExplorer();
  } catch (error) {
    treeRoot.innerHTML = `<p class="message">Unable to load the family tree right now.</p>`;
  }
}

function handleTreeInteraction(event: Event): void {
  const target = event.target as HTMLElement | null;
  if (!target) {
    return;
  }

  const rootButton = target.closest<HTMLButtonElement>("[data-tree-root]");
  if (rootButton) {
    const nextRootId = rootButton.dataset.treeRoot;
    if (nextRootId && nextRootId !== selectedRootId) {
      selectedRootId = nextRootId;
      collapsedBranchIds = new Set<string>();
      renderTreeExplorer();
    }
    return;
  }

  const toggleButton = target.closest<HTMLButtonElement>("[data-tree-toggle]");
  if (toggleButton) {
    const nodeId = toggleButton.dataset.treeToggle;
    if (nodeId) {
      if (collapsedBranchIds.has(nodeId)) {
        collapsedBranchIds.delete(nodeId);
      } else {
        collapsedBranchIds.add(nodeId);
      }
      renderTreeExplorer();
    }
    return;
  }

  const actionButton = target.closest<HTMLButtonElement>("[data-tree-action]");
  if (actionButton) {
    const branch = buildSelectedBranch();
    if (!branch) {
      return;
    }

    const action = actionButton.dataset.treeAction;
    if (action === "expand") {
      collapsedBranchIds = new Set<string>();
    } else if (action === "collapse") {
      collapsedBranchIds = new Set(collectCollapsibleIds(branch));
    }

    renderTreeExplorer();
  }
}

function populatePeopleMaps(people: PersonSummary[]): void {
  peopleById.clear();
  peopleByName.clear();

  people.forEach((person) => {
    peopleById.set(person.id, person);
    peopleByName.set(normalizeName(person.name), person);
  });
}

function renderTreeExplorer(): void {
  if (!treeRoot) {
    return;
  }

  const branch = buildSelectedBranch();
  if (!branch) {
    treeRoot.innerHTML = `<p class="message">Unable to build a descendancy view for this branch.</p>`;
    return;
  }

  const explorer = document.createElement("section");
  explorer.className = "tree-explorer";

  explorer.appendChild(renderExplorerHeader(branch));
  explorer.appendChild(renderRootSwitcher());
  explorer.appendChild(renderTreeCanvas(branch));

  treeRoot.innerHTML = "";
  treeRoot.appendChild(explorer);
}

function renderExplorerHeader(branch: DescendancyNode): HTMLElement {
  const header = document.createElement("div");
  header.className = "tree-explorer-header";

  const copy = document.createElement("div");
  copy.className = "tree-explorer-copy";

  const eyebrow = document.createElement("p");
  eyebrow.className = "subtle";
  eyebrow.textContent = "Descendancy Explorer";
  copy.appendChild(eyebrow);

  const title = document.createElement("h3");
  title.textContent = formatFamilyLabel(branch.person, branch.spouseName);
  copy.appendChild(title);

  const summary = document.createElement("p");
  summary.className = "tree-explorer-summary";
  summary.textContent = `${countGenerations(branch)} generations | ${countPeopleInBranch(branch)} people shown`;
  copy.appendChild(summary);

  const hint = document.createElement("p");
  hint.className = "tree-explorer-hint";
  hint.textContent = "Choose an ancestor line, then expand or collapse each family node to follow descendants.";
  copy.appendChild(hint);

  const actions = document.createElement("div");
  actions.className = "tree-explorer-actions";
  actions.appendChild(createActionButton("Expand all", "expand"));
  actions.appendChild(createActionButton("Collapse all", "collapse"));

  header.appendChild(copy);
  header.appendChild(actions);

  return header;
}

function renderRootSwitcher(): HTMLElement {
  const switcher = document.createElement("div");
  switcher.className = "tree-root-switcher";

  visibleRootIds.forEach((rootId) => {
    const person = peopleById.get(rootId);
    if (!person) {
      return;
    }

    const branch = buildDescendancyBranch(rootId, new Set<string>());
    const button = document.createElement("button");
    button.type = "button";
    button.className = "tree-root-pill";
    if (rootId === selectedRootId) {
      button.classList.add("is-active");
    }
    button.dataset.treeRoot = rootId;

    const title = document.createElement("strong");
    title.textContent = formatFamilyLabel(person, person.spouse);
    button.appendChild(title);

    const meta = document.createElement("span");
    meta.className = "tree-root-meta";
    meta.textContent = `${formatBranchLabel(person.branch)} | ${countGenerations(branch)} generations`;
    button.appendChild(meta);

    switcher.appendChild(button);
  });

  return switcher;
}

function renderTreeCanvas(branch: DescendancyNode): HTMLElement {
  const canvas = document.createElement("div");
  canvas.className = "tree-canvas";
  canvas.appendChild(renderBranch(branch));
  return canvas;
}

function renderBranch(node: DescendancyNode): HTMLElement {
  const branch = document.createElement("section");
  branch.className = "desc-tree-branch";

  const isCollapsed = collapsedBranchIds.has(node.id);
  if (isCollapsed) {
    branch.classList.add("is-collapsed");
  }

  const family = document.createElement("div");
  family.className = "desc-tree-family";

  const controls = document.createElement("div");
  controls.className = "desc-tree-controls";

  if (node.children.length) {
    controls.appendChild(createToggleButton(node.id, node.children.length, isCollapsed));
  } else {
    const leaf = document.createElement("span");
    leaf.className = "desc-tree-leaf";
    leaf.textContent = "No descendants listed";
    controls.appendChild(leaf);
  }

  const couple = document.createElement("div");
  couple.className = "desc-tree-couple";
  couple.appendChild(createPersonCard(node.person, undefined, undefined));

  if (node.spouseName) {
    const connector = document.createElement("div");
    connector.className = "tree-family-connector";
    connector.setAttribute("aria-hidden", "true");
    couple.appendChild(connector);
    couple.appendChild(createPersonCard(node.spouse, node.spouseName, "Spouse"));
  }

  family.appendChild(controls);
  family.appendChild(couple);
  branch.appendChild(family);

  if (node.children.length && !isCollapsed) {
    const children = document.createElement("div");
    children.className = "desc-tree-children";
    node.children.forEach((child) => children.appendChild(renderBranch(child)));
    branch.appendChild(children);
  }

  return branch;
}

function createActionButton(label: string, action: string): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "tree-action-button";
  button.dataset.treeAction = action;
  button.textContent = label;
  return button;
}

function createToggleButton(
  nodeId: string,
  childCount: number,
  isCollapsed: boolean
): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "desc-tree-toggle";
  button.dataset.treeToggle = nodeId;
  button.setAttribute("aria-expanded", String(!isCollapsed));

  const icon = document.createElement("span");
  icon.className = "desc-tree-toggle-icon";
  icon.textContent = isCollapsed ? "+" : "-";
  button.appendChild(icon);

  const label = document.createElement("span");
  label.className = "desc-tree-toggle-label";
  label.textContent = isCollapsed ? "Show descendants" : "Hide descendants";
  button.appendChild(label);

  const count = document.createElement("span");
  count.className = "desc-tree-toggle-count";
  count.textContent = `${childCount} ${childCount === 1 ? "branch" : "branches"}`;
  button.appendChild(count);

  return button;
}

function createPersonCard(
  person: PersonSummary | undefined,
  fallbackName: string | undefined,
  roleLabel: string | undefined
): HTMLElement {
  const resolvedName = person?.name || fallbackName || "Unknown relative";
  const card = person?.id ? document.createElement("a") : document.createElement("div");

  if (person?.id) {
    (card as HTMLAnchorElement).href = `/person?id=${encodeURIComponent(person.id)}`;
  }

  card.className = "tree-person";
  if (!person?.id) {
    card.classList.add("is-static");
  }

  if (roleLabel) {
    const role = document.createElement("span");
    role.className = "tree-person-role";
    role.textContent = roleLabel;
    card.appendChild(role);
  }

  const strong = document.createElement("strong");
  strong.textContent = resolvedName;
  card.appendChild(strong);

  if (person?.branch) {
    const tag = document.createElement("span");
    tag.className = "branch-tag";
    tag.textContent = capitalize(person.branch);
    card.appendChild(tag);
  }

  const years = formatLifeSpan(person?.birthYear, person?.deathYear);
  if (years) {
    const meta = document.createElement("span");
    meta.className = "tree-life-span";
    meta.textContent = years;
    card.appendChild(meta);
  }

  if (!person?.id && roleLabel) {
    const note = document.createElement("span");
    note.className = "tree-life-span";
    note.textContent = "Record not linked yet";
    card.appendChild(note);
  }

  return card;
}

function buildSelectedBranch(): DescendancyNode | null {
  if (!selectedRootId || !peopleById.has(selectedRootId)) {
    return null;
  }

  return buildDescendancyBranch(selectedRootId, new Set<string>());
}

function buildVisibleRoots(people: PersonSummary[]): PersonSummary[] {
  const rootPeople = people.filter((person) => !hasKnownParent(person));
  const rootsById = new Map(rootPeople.map((person) => [person.id, person]));

  return rootPeople
    .filter((person) => !shouldSuppressRoot(person, rootsById))
    .sort(comparePeople);
}

// Treat each spouse pair as one family unit so descendants render once per line.
function buildDescendancyBranch(personId: string, visited: Set<string>): DescendancyNode {
  const person = peopleById.get(personId);
  if (!person) {
    throw new Error(`Missing person record for ${personId}`);
  }

  const spouse = getSpousePerson(person.spouse);
  const nextVisited = new Set(visited);
  nextVisited.add(person.id);

  if (spouse?.id) {
    nextVisited.add(spouse.id);
  }

  const childIds = collectFamilyChildIds(person, spouse).filter((childId) => !nextVisited.has(childId));
  const children = childIds.map((childId) => buildDescendancyBranch(childId, nextVisited));

  return {
    id: person.id,
    person,
    spouse,
    spouseName: person.spouse,
    children,
  };
}

function collectFamilyChildIds(person: PersonSummary, spouse?: PersonSummary): string[] {
  const childIds = new Set<string>(person.children || []);
  (spouse?.children || []).forEach((childId) => childIds.add(childId));

  return Array.from(childIds)
    .filter((childId) => peopleById.has(childId))
    .sort((leftId, rightId) => comparePeopleById(leftId, rightId));
}

function collectCollapsibleIds(node: DescendancyNode): string[] {
  const ids = node.children.length ? [node.id] : [];
  node.children.forEach((child) => ids.push(...collectCollapsibleIds(child)));
  return ids;
}

function countGenerations(node: DescendancyNode): number {
  if (!node.children.length) {
    return 1;
  }

  return 1 + Math.max(...node.children.map((child) => countGenerations(child)));
}

function countPeopleInBranch(node: DescendancyNode, seen = new Set<string>()): number {
  seen.add(node.person.id);

  if (node.spouse?.id) {
    seen.add(node.spouse.id);
  } else if (node.spouseName) {
    seen.add(`spouse:${normalizeName(node.spouseName)}`);
  }

  node.children.forEach((child) => countPeopleInBranch(child, seen));
  return seen.size;
}

function comparePeople(left: PersonSummary, right: PersonSummary): number {
  const leftBirthYear = left.birthYear ?? Number.MAX_SAFE_INTEGER;
  const rightBirthYear = right.birthYear ?? Number.MAX_SAFE_INTEGER;

  if (leftBirthYear !== rightBirthYear) {
    return leftBirthYear - rightBirthYear;
  }

  return left.name.localeCompare(right.name);
}

function comparePeopleById(leftId: string, rightId: string): number {
  const left = peopleById.get(leftId);
  const right = peopleById.get(rightId);

  if (!left || !right) {
    return leftId.localeCompare(rightId);
  }

  return comparePeople(left, right);
}

function shouldSuppressRoot(
  person: PersonSummary,
  rootsById: Map<string, PersonSummary>
): boolean {
  const spouse = getSpousePerson(person.spouse);

  if (!spouse) {
    return false;
  }

  const spouseRoot = rootsById.get(spouse.id);
  if (spouseRoot) {
    return comparePeople(person, spouseRoot) > 0;
  }

  return hasKnownParent(spouse);
}

function hasKnownParent(person: PersonSummary): boolean {
  return (person.parents || []).some((parentId) => peopleById.has(parentId));
}

function getSpousePerson(spouseName: string | undefined): PersonSummary | undefined {
  if (!spouseName) {
    return undefined;
  }

  return peopleByName.get(normalizeName(spouseName));
}

function formatFamilyLabel(person: PersonSummary, spouseName: string | undefined): string {
  if (spouseName) {
    return `${person.name} + ${spouseName}`;
  }

  return person.name;
}

function formatBranchLabel(branch: string | undefined): string {
  if (!branch) {
    return "Family line";
  }

  return `${capitalize(branch)} line`;
}

function normalizeName(value: string): string {
  return value.trim().toLowerCase();
}

function formatLifeSpan(birthYear?: number, deathYear?: number): string {
  if (birthYear !== undefined && deathYear !== undefined) {
    return `${birthYear}-${deathYear}`;
  }

  if (birthYear !== undefined) {
    return `Born ${birthYear}`;
  }

  if (deathYear !== undefined) {
    return `Died ${deathYear}`;
  }

  return "";
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

loadFamilyTree();
