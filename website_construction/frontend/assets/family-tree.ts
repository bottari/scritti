type TreeNode = {
  id: string;
  name: string;
  branch?: string;
  children?: TreeNode[];
};

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

const treeRoot = document.getElementById("family-tree-root");
const peopleById = new Map<string, PersonSummary>();
const peopleByName = new Map<string, PersonSummary>();

async function loadFamilyTree(): Promise<void> {
  if (!treeRoot) {
    return;
  }

  treeRoot.innerHTML = `<p class="message">Loading family tree...</p>`;

  try {
    const [treeResponse, familyResponse] = await Promise.all([
      fetch("/api/tree"),
      fetch("/api/family"),
    ]);

    const tree = (await treeResponse.json()) as TreeNode[];
    const people = (await familyResponse.json()) as PersonSummary[];

    populatePeopleMaps(people);

    if (!tree.length) {
      treeRoot.innerHTML = `<p class="message">No tree data is available yet.</p>`;
      return;
    }

    const forest = document.createElement("div");
    forest.className = "tree-forest";

    const displayRoots = buildVisibleRoots(sortNodes(tree));
    displayRoots.forEach((node) => forest.appendChild(renderBranch(node, true)));

    treeRoot.innerHTML = "";
    treeRoot.appendChild(forest);
  } catch (error) {
    treeRoot.innerHTML = `<p class="message">Unable to load the family tree right now.</p>`;
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

function buildVisibleRoots(nodes: TreeNode[]): TreeNode[] {
  const rootsById = new Map(nodes.map((node) => [node.id, node]));
  return nodes.filter((node) => !shouldSuppressRoot(node, rootsById));
}

function renderBranch(node: TreeNode, isRoot = false): HTMLElement {
  const branch = document.createElement(isRoot ? "section" : "div");
  branch.className = isRoot ? "tree-cluster" : "tree-branch";
  branch.appendChild(renderFamilyUnit(node, isRoot));

  if (node.children && node.children.length) {
    const children = document.createElement("div");
    children.className = "tree-children";

    node.children.forEach((child) => {
      const childWrapper = document.createElement("div");
      childWrapper.className = "tree-child";
      childWrapper.appendChild(renderBranch(child));
      children.appendChild(childWrapper);
    });

    branch.appendChild(children);
  }

  return branch;
}

function renderFamilyUnit(node: TreeNode, isRoot: boolean): HTMLDivElement {
  const family = document.createElement("div");
  family.className = isRoot ? "tree-family tree-family-root" : "tree-family";

  const person = peopleById.get(node.id);
  family.appendChild(
    createPersonCard(node.id, node.name, person?.branch || node.branch || "other", person, false)
  );

  const spouse = getSpousePerson(person?.spouse);
  if (person?.spouse) {
    const connector = document.createElement("div");
    connector.className = "tree-family-connector";
    connector.setAttribute("aria-hidden", "true");
    family.appendChild(connector);
    family.appendChild(
      createPersonCard(
        spouse?.id,
        person.spouse,
        spouse?.branch,
        spouse,
        true
      )
    );
  }

  return family;
}

function createPersonCard(
  id: string | undefined,
  name: string,
  branch: string | undefined,
  person: PersonSummary | undefined,
  isSpouse: boolean
): HTMLElement {
  const card = id ? document.createElement("a") : document.createElement("div");

  if (id) {
    (card as HTMLAnchorElement).href = `/person?id=${encodeURIComponent(id)}`;
  }

  card.className = "tree-person";
  if (!id) {
    card.classList.add("is-static");
  }
  if (isSpouse) {
    card.classList.add("tree-person-spouse");
  }

  if (isSpouse) {
    const role = document.createElement("span");
    role.className = "tree-person-role";
    role.textContent = "Spouse";
    card.appendChild(role);
  }

  const strong = document.createElement("strong");
  strong.textContent = name;
  card.appendChild(strong);

  if (branch) {
    const tag = document.createElement("span");
    tag.className = "branch-tag";
    tag.textContent = capitalize(branch);
    card.appendChild(tag);
  }

  const years = formatLifeSpan(person?.birthYear, person?.deathYear);
  if (years) {
    const meta = document.createElement("span");
    meta.className = "tree-life-span";
    meta.textContent = years;
    card.appendChild(meta);
  }

  return card;
}

function sortNodes(nodes: TreeNode[]): TreeNode[] {
  return [...nodes]
    .map((node) => ({
      ...node,
      children: sortNodes(node.children || []),
    }))
    .sort(compareNodes);
}

function compareNodes(left: TreeNode, right: TreeNode): number {
  const leftBirthYear = peopleById.get(left.id)?.birthYear ?? Number.MAX_SAFE_INTEGER;
  const rightBirthYear = peopleById.get(right.id)?.birthYear ?? Number.MAX_SAFE_INTEGER;

  if (leftBirthYear !== rightBirthYear) {
    return leftBirthYear - rightBirthYear;
  }

  return left.name.localeCompare(right.name);
}

function shouldSuppressRoot(node: TreeNode, rootsById: Map<string, TreeNode>): boolean {
  const person = peopleById.get(node.id);
  const spouse = getSpousePerson(person?.spouse);

  if (!person || !spouse) {
    return false;
  }

  const spouseRoot = rootsById.get(spouse.id);
  if (spouseRoot) {
    return compareNodes(node, spouseRoot) > 0;
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
