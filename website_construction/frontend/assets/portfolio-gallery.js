const portfolioGalleryRoot = document.getElementById("portfolio-gallery-root");
let lightboxElement = null;
let lightboxImage = null;
let lightboxCaption = null;
async function loadPortfolioGallery() {
  if (!portfolioGalleryRoot) {
    return;
  }
  ensureLightbox();
  portfolioGalleryRoot.innerHTML = `<p class="message">Loading gallery...</p>`;
  try {
    const response = await fetch("/api/portfolio/photos");
    const photos = await response.json();
    if (!photos.length) {
      portfolioGalleryRoot.innerHTML = `
        <p class="message">
          No photos have been added yet. Drop image files into
          <code>frontend/data/photos</code> and they will appear here automatically.
        </p>
      `;
      return;
    }
    portfolioGalleryRoot.innerHTML = photos
      .map(
        (photo) => `
          <button
            class="gallery-card"
            type="button"
            data-photo-url="${photo.url}"
            data-photo-name="${escapeHtml(photo.name)}"
            data-photo-filename="${escapeHtml(photo.filename)}"
          >
            <img src="${photo.url}" alt="${photo.name}" loading="lazy" />
            <span class="gallery-card-copy">
              <strong>${photo.name}</strong>
              <span>${photo.filename}</span>
            </span>
          </button>
        `
      )
      .join("");
    portfolioGalleryRoot.querySelectorAll(".gallery-card").forEach((card) => {
      card.addEventListener("click", () => {
        openLightbox(card.dataset.photoUrl || "", card.dataset.photoName || "Photo", card.dataset.photoFilename || "");
      });
    });
  } catch (error) {
    portfolioGalleryRoot.innerHTML = `<p class="message">Unable to load the portfolio gallery right now.</p>`;
  }
}
function ensureLightbox() {
  if (lightboxElement) {
    return;
  }
  lightboxElement = document.createElement("div");
  lightboxElement.className = "lightbox";
  lightboxElement.setAttribute("aria-hidden", "true");
  lightboxElement.innerHTML = `
    <div class="lightbox-backdrop" data-lightbox-close="true"></div>
    <div class="lightbox-dialog" role="dialog" aria-modal="true" aria-label="Photo viewer">
      <button class="lightbox-close" type="button" aria-label="Close photo viewer">Close</button>
      <img class="lightbox-image" alt="" />
      <p class="lightbox-caption"></p>
    </div>
  `;
  document.body.appendChild(lightboxElement);
  lightboxImage = lightboxElement.querySelector(".lightbox-image");
  lightboxCaption = lightboxElement.querySelector(".lightbox-caption");
  lightboxElement.addEventListener("click", (event) => {
    const target = event.target;
    if (target.dataset.lightboxClose === "true" || target.classList.contains("lightbox-close")) {
      closeLightbox();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeLightbox();
    }
  });
}
function openLightbox(url, name, filename) {
  if (!lightboxElement || !lightboxImage || !lightboxCaption) {
    return;
  }
  lightboxImage.src = url;
  lightboxImage.alt = name;
  lightboxCaption.textContent = filename ? `${name} - ${filename}` : name;
  lightboxElement.classList.add("is-open");
  lightboxElement.setAttribute("aria-hidden", "false");
  document.body.classList.add("lightbox-open");
}
function closeLightbox() {
  if (!lightboxElement || !lightboxImage || !lightboxCaption) {
    return;
  }
  lightboxElement.classList.remove("is-open");
  lightboxElement.setAttribute("aria-hidden", "true");
  lightboxImage.src = "";
  lightboxImage.alt = "";
  lightboxCaption.textContent = "";
  document.body.classList.remove("lightbox-open");
}
function escapeHtml(value) {
  return value.replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}
loadPortfolioGallery();
