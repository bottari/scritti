type PortfolioMediaType = "image" | "video";

type PortfolioMediaEntry = {
  name: string;
  filename: string;
  url: string;
  mediaType: PortfolioMediaType;
};

type PortfolioLayoutVariant = "featured" | "wide" | "tall" | "standard" | "compact";

const LAYOUT_VARIANT_CYCLE: PortfolioLayoutVariant[] = [
  "featured",
  "standard",
  "compact",
  "wide",
  "tall",
  "compact",
  "standard",
  "wide",
  "standard",
  "tall",
  "compact",
  "wide",
];

const SMALL_GALLERY_VARIANTS: Record<number, PortfolioLayoutVariant[]> = {
  1: ["featured"],
  2: ["featured", "wide"],
  3: ["featured", "standard", "compact"],
  4: ["featured", "standard", "compact", "wide"],
};

const portfolioGalleryRoot = document.getElementById("portfolio-gallery-root");
let lightboxElement: HTMLDivElement | null = null;
let lightboxImage: HTMLImageElement | null = null;
let lightboxVideo: HTMLVideoElement | null = null;
let lightboxCaption: HTMLParagraphElement | null = null;

async function loadPortfolioGallery(): Promise<void> {
  if (!portfolioGalleryRoot) {
    return;
  }

  ensureLightbox();

  portfolioGalleryRoot.dataset.itemCount = "0";
  portfolioGalleryRoot.innerHTML = `<p class="message">Loading gallery...</p>`;

  try {
    const response = await fetch("/api/portfolio/photos");
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const mediaEntries = (await response.json()) as PortfolioMediaEntry[];

    if (!mediaEntries.length) {
      portfolioGalleryRoot.innerHTML = `
        <p class="message">
          No photos or videos have been added yet. Drop image files or MP4 videos into
          <code>frontend/data/photos</code> and they will appear here automatically.
        </p>
      `;
      return;
    }

    portfolioGalleryRoot.dataset.itemCount = String(mediaEntries.length);
    portfolioGalleryRoot.innerHTML = mediaEntries
      .map((mediaEntry, index) => renderGalleryItem(mediaEntry, index, mediaEntries.length))
      .join("");

    portfolioGalleryRoot.querySelectorAll<HTMLButtonElement>(".portfolio-item").forEach((card) => {
      card.addEventListener("click", () => {
        openLightbox(
          card.dataset.mediaUrl || "",
          card.dataset.mediaName || "Media",
          card.dataset.mediaFilename || "",
          toMediaType(card.dataset.mediaType)
        );
      });
    });
  } catch (error) {
    portfolioGalleryRoot.innerHTML = `<p class="message">Unable to load the portfolio gallery right now.</p>`;
  }
}

function renderGalleryItem(
  mediaEntry: PortfolioMediaEntry,
  index: number,
  totalItems: number
): string {
  const layoutVariant = getLayoutVariant(index, totalItems, mediaEntry.mediaType);
  const mediaTypeClass = mediaEntry.mediaType === "video" ? "portfolio-item--video" : "portfolio-item--image";

  return `
    <button
      class="portfolio-item ${mediaTypeClass} portfolio-item--${layoutVariant}"
      type="button"
      aria-label="${escapeHtml(mediaEntry.name)}"
      data-media-url="${escapeHtml(mediaEntry.url)}"
      data-media-name="${escapeHtml(mediaEntry.name)}"
      data-media-filename="${escapeHtml(mediaEntry.filename)}"
      data-media-type="${mediaEntry.mediaType}"
    >
      <span class="portfolio-item__media ${mediaEntry.mediaType === "video" ? "is-video" : ""}">
        ${renderGalleryPreview(mediaEntry, index)}
        ${mediaEntry.mediaType === "video" ? `<span class="portfolio-item__play-icon" aria-hidden="true"></span>` : ""}
      </span>
    </button>
  `;
}

function renderGalleryPreview(mediaEntry: PortfolioMediaEntry, index: number): string {
  if (mediaEntry.mediaType === "video") {
    return `
      <video
        src="${escapeHtml(`${mediaEntry.url}#t=0.1`)}"
        muted
        playsinline
        preload="metadata"
        aria-hidden="true"
      ></video>
    `;
  }

  const loadingStrategy = index < 3 ? "eager" : "lazy";
  const fetchPriority = index === 0 ? ` fetchpriority="high"` : "";

  return `
    <img
      src="${escapeHtml(mediaEntry.url)}"
      alt="${escapeHtml(mediaEntry.name)}"
      loading="${loadingStrategy}"
      decoding="async"${fetchPriority}
    />
  `;
}

function getLayoutVariant(
  index: number,
  totalItems: number,
  mediaType: PortfolioMediaType
): PortfolioLayoutVariant {
  const variantSource = SMALL_GALLERY_VARIANTS[totalItems] || LAYOUT_VARIANT_CYCLE;
  const preferredVariant = variantSource[index % variantSource.length] || "standard";
  return normalizeLayoutVariant(preferredVariant, mediaType);
}

function normalizeLayoutVariant(
  variant: PortfolioLayoutVariant,
  mediaType: PortfolioMediaType
): PortfolioLayoutVariant {
  if (mediaType !== "video") {
    return variant;
  }

  if (variant === "tall") {
    return "wide";
  }

  return variant;
}

function ensureLightbox(): void {
  if (lightboxElement) {
    return;
  }

  lightboxElement = document.createElement("div");
  lightboxElement.className = "lightbox";
  lightboxElement.setAttribute("aria-hidden", "true");
  lightboxElement.innerHTML = `
    <div class="lightbox-backdrop" data-lightbox-close="true"></div>
    <div class="lightbox-dialog" role="dialog" aria-modal="true" aria-label="Portfolio media viewer">
      <button class="lightbox-close" type="button" aria-label="Close media viewer">Close</button>
      <div class="lightbox-media">
        <img class="lightbox-image" alt="" hidden />
        <video class="lightbox-video" controls playsinline preload="metadata" hidden></video>
      </div>
      <p class="lightbox-caption"></p>
    </div>
  `;
  document.body.appendChild(lightboxElement);

  lightboxImage = lightboxElement.querySelector(".lightbox-image");
  lightboxVideo = lightboxElement.querySelector(".lightbox-video");
  lightboxCaption = lightboxElement.querySelector(".lightbox-caption");

  lightboxElement.addEventListener("click", (event) => {
    const target = event.target as HTMLElement;
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

function openLightbox(
  url: string,
  name: string,
  filename: string,
  mediaType: PortfolioMediaType
): void {
  if (!lightboxElement || !lightboxImage || !lightboxVideo || !lightboxCaption) {
    return;
  }

  if (mediaType === "video") {
    lightboxImage.hidden = true;
    lightboxImage.src = "";
    lightboxImage.alt = "";

    lightboxVideo.hidden = false;
    lightboxVideo.src = url;
    lightboxVideo.load();
  } else {
    lightboxVideo.pause();
    lightboxVideo.hidden = true;
    lightboxVideo.removeAttribute("src");
    lightboxVideo.load();

    lightboxImage.hidden = false;
    lightboxImage.src = url;
    lightboxImage.alt = name;
  }

  lightboxCaption.textContent = filename ? `${name} - ${filename}` : name;
  lightboxElement.classList.add("is-open");
  lightboxElement.setAttribute("aria-hidden", "false");
  document.body.classList.add("lightbox-open");
}

function closeLightbox(): void {
  if (!lightboxElement || !lightboxImage || !lightboxVideo || !lightboxCaption) {
    return;
  }

  lightboxElement.classList.remove("is-open");
  lightboxElement.setAttribute("aria-hidden", "true");
  lightboxImage.src = "";
  lightboxImage.alt = "";
  lightboxImage.hidden = true;
  lightboxVideo.pause();
  lightboxVideo.hidden = true;
  lightboxVideo.removeAttribute("src");
  lightboxVideo.load();
  lightboxCaption.textContent = "";
  document.body.classList.remove("lightbox-open");
}

function toMediaType(value: string | undefined): PortfolioMediaType {
  return value === "video" ? "video" : "image";
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

loadPortfolioGallery();
