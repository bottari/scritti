const treeRoot = document.getElementById("family-tree-root");
async function loadFamilyTree() {
  if (!treeRoot) {
    return;
  }
  treeRoot.innerHTML = `<p class="message">Loading family tree…</p>`;
  try {
    const response = await fetch("/api/tree");
    const tree = await response.json();
    if (!tree.length) {
      treeRoot.innerHTML = `<p class="message">No tree data is available yet.</p>`;
      return;
    }
    const wrapper = document.createElement("ul");
    tree.forEach((node) => wrapper.appendChild(renderNode(node)));
    treeRoot.innerHTML = "";
    treeRoot.appendChild(wrapper);
  } catch (error) {
    treeRoot.innerHTML = `<p class="message">Unable to load the family tree right now.</p>`;
  }
}
function renderNode(node) {
  const item = document.createElement("li");
  const link = document.createElement("a");
  link.href = `/person?id=${encodeURIComponent(node.id)}`;
  link.className = "tree-node";
  link.innerHTML = `
    <strong>${node.name}</strong>
    <span class="branch-tag">${capitalize(node.branch || "other")}</span>
  `;
  item.appendChild(link);
  if (node.children && node.children.length) {
    const childList = document.createElement("ul");
    node.children.forEach((child) => childList.appendChild(renderNode(child)));
    item.appendChild(childList);
  }
  return item;
}
function capitalize(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
loadFamilyTree();
