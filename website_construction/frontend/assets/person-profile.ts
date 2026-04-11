type PersonDetail = {
  id: string;
  name: string;
  birthYear?: number;
  deathYear?: number;
  bio?: string;
  parents?: string[];
  children?: string[];
  siblings?: string[];
  branch?: string;
};

type PersonLookup = {
  id: string;
  name: string;
};

const profileRoot = document.getElementById("person-profile-root");

async function loadPersonProfile(): Promise<void> {
  if (!profileRoot) {
    return;
  }

  const personId = new URLSearchParams(window.location.search).get("id");
  if (!personId) {
    profileRoot.innerHTML = `<p class="message">No person id was provided.</p>`;
    return;
  }

  profileRoot.innerHTML = `<p class="message">Loading profile…</p>`;

  try {
    const [personResponse, familyResponse] = await Promise.all([
      fetch(`/api/person/${encodeURIComponent(personId)}`),
      fetch("/api/family"),
    ]);

    if (!personResponse.ok) {
      throw new Error("Unable to find person");
    }

    const person = (await personResponse.json()) as PersonDetail;
    const family = (await familyResponse.json()) as PersonLookup[];
    const peopleById = new Map(family.map((entry) => [entry.id, entry.name]));

    profileRoot.innerHTML = `
      <article class="detail-card">
        <span class="branch-tag">${capitalize(person.branch || "other")}</span>
        <h2>${person.name}</h2>
        <p>${formatYears(person.birthYear, person.deathYear)}</p>
        <p>${person.bio || "No biography has been added yet."}</p>
      </article>
      <div class="profile-grid">
        <section class="detail-card">
          <h3>Parents</h3>
          <div class="link-list">${renderPersonLinks(person.parents, peopleById)}</div>
        </section>
        <section class="detail-card">
          <h3>Children</h3>
          <div class="link-list">${renderPersonLinks(person.children, peopleById)}</div>
        </section>
        <section class="detail-card">
          <h3>Siblings</h3>
          <div class="link-list">${renderPersonLinks(person.siblings, peopleById)}</div>
        </section>
        <section class="detail-card">
          <h3>Record</h3>
          <p><strong>Identifier:</strong> ${person.id}</p>
          <p><strong>Branch:</strong> ${capitalize(person.branch || "other")}</p>
        </section>
      </div>
    `;
  } catch (error) {
    profileRoot.innerHTML = `<p class="message">Unable to load this person profile right now.</p>`;
  }
}

function renderPersonLinks(ids: string[] = [], peopleById: Map<string, string>): string {
  if (!ids.length) {
    return `<span>No records available.</span>`;
  }

  return ids
    .map((id) => {
      const name = peopleById.get(id) || id;
      return `<a class="person-link" href="/person?id=${encodeURIComponent(id)}">${name}</a>`;
    })
    .join("");
}

function formatYears(birthYear?: number, deathYear?: number): string {
  if (birthYear && deathYear) {
    return `${birthYear} – ${deathYear}`;
  }
  if (birthYear) {
    return `Born ${birthYear}`;
  }
  if (deathYear) {
    return `Died ${deathYear}`;
  }
  return "Birth and death details not available";
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

loadPersonProfile();
