const page = document.body.dataset.page;
document.querySelectorAll(".top-nav a").forEach((link) => {
  const href = link.getAttribute("href");
  if (
    (page === "home" && (href === "/" || href === "/home")) ||
    (page === "about" && href === "/about") ||
    (page === "portfolio" && href === "/portfolio") ||
    (page === "family" && (href === "/family" || href === "/person"))
  ) {
    link.classList.add("is-active");
  }
});
