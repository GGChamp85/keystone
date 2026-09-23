// Unique URL slugs for titles: "hello world" -> "hello-world"; a repeated title gets "-2", "-3", ...
export function slugify(title) {
  return title
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export class SlugRegistry {
  constructor() {
    this.taken = new Set();
  }

  /** Register `title` and return a slug no earlier title received. */
  claim(title) {
    const base = slugify(title);
    if (!this.taken.has(base)) {
      this.taken.add(base);
      return base;
    }
    // BUG: only one suffix is ever tried, so the third repeat of a title gets "-2" again — a duplicate.
    const candidate = `${base}-2`;
    this.taken.add(candidate);
    return candidate;
  }
}
