"""
Created on Jan 29, 2026.
radiopaedia_database.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import os
import time
import random
import hashlib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List
import csv
import pandas as pd
import xml.etree.ElementTree as ET


import warnings
warnings.filterwarnings('ignore')




class RadiopaediaArticleURLExtractor:
    BASE = "https://radiopaedia.org"

    SITEMAP_CANDIDATES = [
        f"{BASE}/sitemap?lang=us",   # advertised in robots.txt
        f"{BASE}/sitemap.xml",
        f"{BASE}/sitemap_index.xml",
        f"{BASE}/sitemap-index.xml",
    ]

    # Heuristic: real article pages are /articles/<slug>, not /articles (index) or /articles/section/...
    ARTICLE_URL_RE = re.compile(
        r"^https?://(?:www\.)?radiopaedia\.org/articles/[a-z0-9][a-z0-9\-]*$",
        re.I,
    )

    def __init__(self, output_path, sleep_s: float = 0.3):
        self.output_path = output_path
        self.sleep_s = sleep_s
        self.session = requests.Session()
        self.session.headers.update({
            # Use a normal browser UA. Do NOT impersonate search-engine bots.
            "User-Agent": "Mozilla/5.0 (compatible; radiopaedia-sitemap-fetcher/1.0; +https://example.invalid)"
        })

    def fetch(self, url: str, sleep_s: float | None = None) -> bytes:
        r = self.session.get(url, timeout=30)
        r.raise_for_status()
        time.sleep(self.sleep_s if sleep_s is None else sleep_s)  # be gentle
        return r.content

    def parse_xml(self, content: bytes) -> ET.Element:
        return ET.fromstring(content)

    def ns(self, tag: str, root: ET.Element) -> str:
        # Handle default xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        if root.tag.startswith("{"):
            uri = root.tag.split("}")[0].strip("{")
            return f"{{{uri}}}{tag}"
        return tag

    def is_article_sitemap_loc(self, loc: str) -> bool:
        # common naming: sitemap-articles_*.xml (seen in the wild)
        return "sitemap-articles" in loc

    def discover_sitemap_index(self) -> str:
        last_err = None
        for u in self.SITEMAP_CANDIDATES:
            try:
                self.fetch(u, sleep_s=0.0)
                return u
            except Exception as e:
                last_err = e

        # Fallback: probe numbered article sitemaps directly (1..200)
        # Stop after a few consecutive 404s.
        misses = 0
        for i in range(1, 201):
            u = f"{self.BASE}/sitemap-articles_{i}.xml"
            try:
                self.fetch(u, sleep_s=0.0)
                return u  # not an index, but we found a sitemap we can start from
            except requests.HTTPError as e:
                if getattr(e.response, "status_code", None) == 404:
                    misses += 1
                    if misses >= 5:
                        break
                else:
                    last_err = e
            except Exception as e:
                last_err = e

        raise RuntimeError(f"Could not access Radiopaedia sitemap endpoints. Last error: {last_err}")

    def extract_article_urls_from_sitemap(self, sitemap_url: str) -> set[str]:
        content = self.fetch(sitemap_url)
        root = self.parse_xml(content)

        urls: set[str] = set()

        # sitemapindex
        if root.tag.endswith("sitemapindex"):
            for sm in root.findall(self.ns("sitemap", root)):
                loc_el = sm.find(self.ns("loc", root))
                if loc_el is None or not loc_el.text:
                    continue
                loc = loc_el.text.strip()
                if self.is_article_sitemap_loc(loc):
                    urls |= self.extract_article_urls_from_sitemap(loc)
            return urls

        # urlset
        if root.tag.endswith("urlset"):
            for u in root.findall(self.ns("url", root)):
                loc_el = u.find(self.ns("loc", root))
                if loc_el is None or not loc_el.text:
                    continue
                loc = loc_el.text.strip()
                if self.ARTICLE_URL_RE.match(loc):
                    urls.add(loc)
            return urls

        raise ValueError(f"Unrecognized sitemap XML root tag: {root.tag}")

    def run(self) -> None:
        entry = self.discover_sitemap_index()
        urls = self.extract_article_urls_from_sitemap(entry)

        # If entry was a single sitemap-articles_N.xml (not an index), also try nearby numbers
        if "sitemap-articles_" in entry and entry.endswith(".xml"):
            parsed = urlparse(entry)
            m = re.search(r"sitemap-articles_(\d+)\.xml$", parsed.path)
            if m:
                start_i = int(m.group(1))
                # Try a reasonable window around it; adjust if needed.
                for i in range(max(1, start_i - 50), start_i + 51):
                    u = f"{self.BASE}/sitemap-articles_{i}.xml"
                    if u == entry:
                        continue
                    try:
                        urls |= self.extract_article_urls_from_sitemap(u)
                    except requests.HTTPError as e:
                        if getattr(e.response, "status_code", None) == 404:
                            continue
                        raise

        urls = sorted(urls)
        print(f"Found {len(urls)} article URLs")

        with open(self.output_path, "w", encoding="utf-8") as f:
            for u in urls:
                f.write(u + "\n")

        print(f"Wrote: {self.output_path}")






@dataclass
class RadiopaediaDownloadConfig:
    out_dir: str
    url_list_path: str
    timeout_sec: int = 100
    base_delay_sec: float = 12.0
    jitter_sec: float = 6.0
    masterlist_csv_path: str | None = None


class RadiopaediaArticleDownloader:
    """
    Minimal, browser-faithful Radiopaedia article downloader.
    The authors of this paper have a permission to download this way.
    Any other person who wants to download this, must first get the right permission from Radiopaedia.
    """

    def __init__(self, cfg: RadiopaediaDownloadConfig) -> None:
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html",
        })

    @staticmethod
    def _is_valid_article_url(url: str) -> bool:
        try:
            u = urlparse(url.strip())
            return (
                u.scheme in {"http", "https"}
                and u.netloc.endswith("radiopaedia.org")
                and u.path.startswith("/articles/")
            )
        except Exception:
            return False

    @staticmethod
    def _slugify(url: str) -> str:
        h = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
        slug = url.rstrip("/").split("/")[-1]
        slug = "".join(c for c in slug if c.isalnum() or c in {"-", "_"})
        return f"{slug}__{h}"

    def _sleep(self) -> None:
        time.sleep(self.cfg.base_delay_sec + random.uniform(0, self.cfg.jitter_sec))

    def _fetch_text(self, url: str) -> str:
        resp = self.session.get(url, timeout=self.cfg.timeout_sec)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        return text

    def download_all(self) -> None:
        os.makedirs(self.cfg.out_dir, exist_ok=True)

        with open(self.cfg.url_list_path, "r", encoding="utf-8") as f:
            urls = [u.strip() for u in f if u.strip()]

        urls = sorted({u for u in urls if self._is_valid_article_url(u)})

        masterlist_rows: list[dict] = []
        masterlist_path = self.cfg.masterlist_csv_path or os.path.join(self.cfg.out_dir, "masterlist.csv")

        print(f"Will download {len(urls)} articles into {self.cfg.out_dir}")

        for i, url in enumerate(urls, 1):
            name = self._slugify(url)
            rel_raw_txt_path = f"{name}.txt"
            path = os.path.join(self.cfg.out_dir, rel_raw_txt_path)

            status = None
            error = ""
            bytes_written = 0
            skipped_existing = False

            if os.path.exists(path) and os.path.getsize(path) > 0:
                skipped_existing = True
                bytes_written = int(os.path.getsize(path))
                status = 200
                masterlist_rows.append({
                    "url": url,
                    "status": status,
                    "error": error,
                    "rel_raw_txt_path": rel_raw_txt_path,
                    "bytes": bytes_written,
                    "skipped_existing": skipped_existing,
                })
                continue

            print(f"[{i}/{len(urls)}] GET {url}")

            try:
                text = self._fetch_text(url)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                bytes_written = int(os.path.getsize(path)) if os.path.exists(path) else 0
                status = 200
            except Exception as e:
                print("  ERROR:", e)
                error = str(e)
                try:
                    status = int(getattr(getattr(e, "response", None), "status_code", -1))
                except Exception:
                    status = -1

            masterlist_rows.append({
                "url": url,
                "status": status if status is not None else -1,
                "error": error,
                "rel_raw_txt_path": rel_raw_txt_path,
                "bytes": bytes_written,
                "skipped_existing": skipped_existing,
            })

            self._sleep()

        df = pd.DataFrame(masterlist_rows, columns=[
            "url", "status", "error", "rel_raw_txt_path", "bytes", "skipped_existing"
        ])
        os.makedirs(os.path.dirname(masterlist_path) or ".", exist_ok=True)
        df.to_csv(masterlist_path, index=False, encoding="utf-8")
        print(f"Wrote masterlist: {masterlist_path}")









@dataclass
class RadiopaediaPostprocessConfig:
    masterlist_csv: str
    raw_base_dir: str
    out_dir_clean: str
    out_masterlist_csv: str
    min_clean_chars: int = 800
    overwrite: bool = False
    keep_references: bool = True        # keep "References" section text if present


class RadiopaediaArticlePostprocessor:
    """
    Postprocess downloaded Radiopaedia text dumps into cleaned article-body text.

    Expected masterlist columns:
      - url
      - status
      - rel_raw_txt_path   (relative to cfg.raw_base_dir)

    Writes:
      - cleaned txts under cfg.out_dir_clean
      - updated CSV to cfg.out_masterlist_csv with:
          rel_clean_txt_path, title, clean_chars, clean_lines, clean_ok, clean_error
    """

    def __init__(self, cfg: RadiopaediaPostprocessConfig) -> None:
        self.cfg = cfg
        self.raw_base_dir = Path(cfg.raw_base_dir)
        self.out_dir_clean = Path(cfg.out_dir_clean)
        self.out_dir_clean.mkdir(parents=True, exist_ok=True)

        # Exact junk lines often present (nav/footer)
        self._drop_exact = {
            "Radiopaedia.org",
            "Search",
            "Sign up",
            "Log in",
            "Log In",
            "Sign Up",
            "Menu",
            "About",
            "Articles",
            "Cases",
            "Courses",
            "Questions",
            "Quiz",
            "Recent Edits",
            "Supporters",
            "Donate",
            "Jobs",
            "Blog",
            "Privacy",
            "Terms",
            "Contact",
            "Help",
            "Cite",
            "Cite this article",
            "Edit article",
            "Share",
            "Download",
            "Print",
            "Feedback",
            "Donate to Radiopaedia",
            "Start a new discussion",
            "Accept",
            "ADVERTISEMENT: Radiopaedia is free thanks to our supporters and advertisers.",
            "Become a Gold Supporter and see no third-party ads.",
            "×",
            "/",
            ".",
        }

        # Drop lines containing these patterns (ads/social/cookies/etc)
        self._drop_contains_re = re.compile(
            r"(cookie|cookies|privacy policy|terms of use|copyright|all rights reserved|"
            r"donate|log in|sign up|subscribe|newsletter|advert|advertisement|sponsor|"
            r"open in|share this|twitter|facebook|linkedin|instagram|"
            r"do not share my personal information|contact us|licensing|developers|"
            r"radiopaedia team|editorial board|supporters|gold supporter)",
            re.IGNORECASE,
        )

        # Nav markers present in dumps
        self._onpage_marker = "On this page:"
        self._article_marker = "Article:"
        self._images_marker = "Images:"

        # Stop headings that indicate we're past the body (BUT these appear in nav too!)
        self._stop_headings = {
            "Incoming Links",
            "Incoming links",
            "Related articles",
            "Related Articles",
            "Cases and figures",
            "Quiz questions",
            "Promoted articles",
            "By Section:",
            "By System:",
            "Not Applicable",
        }

        # Title line sometimes looks like: "<title> | Radiology Reference Article | Radiopaedia.org"
        self._title_suffix_re = re.compile(r"\s*\|\s*Radiology Reference Article.*$", re.IGNORECASE)

        # Pure reference number line
        self._refnum_re = re.compile(r"^\s*\d+\s*$")

        # Big JSON blocks show up as one giant line starting with "{"
        self._looks_like_json_re = re.compile(r"^\s*[\{\[]")

        # Footer/cookie tail signals
        self._footer_re = re.compile(r"^©\s*\d{4}", re.IGNORECASE)
        self._cookie_tail_re = re.compile(r"(This site is for use by|To continue you must accept|Terms of Use)", re.IGNORECASE)

    @staticmethod
    def _read_text(path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    @staticmethod
    def _safe_slug_from_filename(rel_raw_txt_path: str) -> str:
        base = Path(rel_raw_txt_path).name
        return base[:-4] if base.endswith(".txt") else base

    @staticmethod
    def _split_nonempty_lines(text: str) -> List[str]:
        return [ln.strip() for ln in text.splitlines() if ln.strip()]

    def _extract_title(self, lines: List[str]) -> str:
        for ln in lines[:80]:
            if "Radiology Reference Article" in ln:
                t = self._title_suffix_re.sub("", ln).strip()
                if t and len(t) >= 3:
                    return t
        for ln in lines[:80]:
            if len(ln) >= 5 and ln not in self._drop_exact and not self._drop_contains_re.search(ln):
                return ln.strip()
        return ""

    def _locate_nav(self, lines: List[str]) -> Dict[str, int | None]:
        idx_onpage = None
        idx_article = None
        idx_images = None

        for i, ln in enumerate(lines[:450]):
            if idx_onpage is None and ln == self._onpage_marker:
                idx_onpage = i
            if idx_article is None and ln == self._article_marker:
                idx_article = i
            if idx_images is None and ln == self._images_marker:
                idx_images = i
                break

        # define a conservative nav_end boundary
        nav_end = None
        if idx_images is not None:
            nav_end = idx_images + 1
        elif idx_onpage is not None:
            nav_end = min(len(lines), idx_onpage + 260)

        return {
            "idx_onpage": idx_onpage,
            "idx_article": idx_article,
            "idx_images": idx_images,
            "nav_end": nav_end,
        }

    def _build_heading_set_from_nav(self, lines: List[str], idx_article: int | None, idx_images: int | None) -> set[str]:
        heading_set: set[str] = set()
        if idx_article is not None:
            nav_end = idx_images if idx_images is not None else min(len(lines), idx_article + 260)
            for ln in lines[idx_article + 1 : nav_end]:
                if not ln or ln.endswith(":") or len(ln) <= 2:
                    continue
                if ln in self._drop_exact:
                    continue
                if self._drop_contains_re.search(ln):
                    continue
                heading_set.add(ln)

        # common headings
        heading_set |= {
            "Terminology",
            "Epidemiology",
            "Risk factors",
            "Protective factors",
            "Pathology",
            "Classification",
            "Immunophenotype",
            "Staging",
            "Clinical presentation",
            "Radiographic appearance",
            "Radiographic features",
            "Mammography",
            "Ultrasound",
            "MRI",
            "CT",
            "Nuclear medicine",
            "Treatment and prognosis",
            "Differential diagnosis",
            "Practical points",
            "Indications",
            "Patient position",
            "Technical factors",
            "Image technical evaluation",
            "History and etymology",
            "References",
        }
        return heading_set

    def _find_body_start(self, lines: List[str], heading_set: set[str], nav_end: int | None) -> int:
        if not lines:
            return 0

        # many "stop headings" appear inside the nav, so we start scanning *after* the nav.
        start_scan = nav_end if nav_end is not None else 0

        def looks_like_prose(s: str) -> bool:
            if not s or s.endswith(":"):
                return False
            if self._refnum_re.match(s):
                return False
            if self._looks_like_json_re.match(s):
                return False
            # simple prose rule
            if len(s.split()) >= 6:
                return True
            return ("." in s) or ("," in s) or (";" in s)

        # Prefer: a heading followed by prose, after nav_end
        for i in range(start_scan, len(lines) - 1):
            ln = lines[i]
            nxt = lines[i + 1]
            if ln in heading_set and looks_like_prose(nxt) and (nxt not in heading_set):
                return i

        # Fallback: first prose-like line after nav_end that isn't junk
        for i in range(start_scan, len(lines)):
            ln = lines[i]
            if ln in self._drop_exact:
                continue
            if self._drop_contains_re.search(ln):
                continue
            if self._looks_like_json_re.match(ln):
                continue
            if len(ln.split()) >= 6:
                return i

        return start_scan

    def _find_body_end(self, lines: List[str], start_idx: int, nav_end: int | None) -> int:
        stop = len(lines)
        min_stop_i = start_idx + 20

        for i in range(start_idx, len(lines)):
            ln = lines[i]

            # hard footer stop
            if self._footer_re.search(ln) or self._cookie_tail_re.search(ln):
                stop = i
                break

            # stop headings ONLY if they occur well after nav and after some body content
            if i >= min_stop_i:
                if nav_end is not None and i < nav_end + 5:
                    pass
                else:
                    if ln in self._stop_headings:
                        stop = i
                        break

            # references stop option
            if (not self.cfg.keep_references) and ln == "References" and i >= min_stop_i:
                stop = i
                break

        return stop

    def _clean_body_lines(self, body_lines: List[str]) -> List[str]:
        out: List[str] = []
        for ln in body_lines:
            if not ln:
                continue
            if ln in self._drop_exact:
                continue
            if self._drop_contains_re.search(ln):
                continue
            if self._refnum_re.match(ln):
                continue
            if self._looks_like_json_re.match(ln):
                continue
            if ln in {self._onpage_marker, self._article_marker, self._images_marker, "Tags:", "Synonyms:", "Read more"}:
                continue
            out.append(ln)

        # de-dup consecutive duplicates
        dedup: List[str] = []
        prev = None
        for ln in out:
            if ln == prev:
                continue
            dedup.append(ln)
            prev = ln
        return dedup

    def _format_clean_text(self, title: str, lines: List[str]) -> str:
        body = "\n".join(lines).strip()
        if title:
            return title.strip() + "\n\n" + body + "\n"
        return body + "\n"

    def postprocess_one(self, rel_raw_txt_path: str) -> Dict[str, object]:
        raw_path = self.raw_base_dir / rel_raw_txt_path

        if not raw_path.exists():
            return {
                "clean_ok": False,
                "error": f"missing_raw_file: {raw_path}",
                "rel_clean_txt_path": "",
                "title": "",
                "clean_chars": 0,
                "clean_lines": 0,
            }

        raw = self._read_text(raw_path)
        lines = self._split_nonempty_lines(raw)
        title = self._extract_title(lines)

        nav = self._locate_nav(lines)
        heading_set = self._build_heading_set_from_nav(lines, nav["idx_article"], nav["idx_images"])

        start_idx = self._find_body_start(lines, heading_set, nav["nav_end"])
        end_idx = self._find_body_end(lines, start_idx, nav["nav_end"])

        body = lines[start_idx:end_idx]
        body = self._clean_body_lines(body)

        # If still tiny, fallback to after nav_end until footer, then clean
        clean_text = self._format_clean_text(title, body)
        if len(clean_text) < self.cfg.min_clean_chars:
            fallback_start = nav["nav_end"] if nav["nav_end"] is not None else 0
            fallback_end = self._find_body_end(lines, fallback_start, nav["nav_end"])
            body2 = self._clean_body_lines(lines[fallback_start:fallback_end])
            clean_text2 = self._format_clean_text(title, body2)
            if len(clean_text2) > len(clean_text):
                body = body2
                clean_text = clean_text2

        if len(clean_text) < self.cfg.min_clean_chars:
            return {
                "clean_ok": False,
                "error": "clean_too_small",
                "rel_clean_txt_path": "",
                "title": title,
                "clean_chars": len(clean_text),
                "clean_lines": len(body),
            }

        slug = self._safe_slug_from_filename(rel_raw_txt_path)
        clean_name = f"{slug}.clean.txt"
        clean_path = self.out_dir_clean / clean_name
        rel_clean = str(Path(clean_name))

        if clean_path.exists() and (not self.cfg.overwrite):
            existing = self._read_text(clean_path)
            return {
                "clean_ok": True,
                "error": "",
                "rel_clean_txt_path": rel_clean,
                "title": title,
                "clean_chars": len(existing),
                "clean_lines": len(self._split_nonempty_lines(existing)),
            }

        self._write_text(clean_path, clean_text)
        return {
            "clean_ok": True,
            "error": "",
            "rel_clean_txt_path": rel_clean,
            "title": title,
            "clean_chars": len(clean_text),
            "clean_lines": len(body),
        }

    @staticmethod
    def _status_is_success(x) -> bool:
        try:
            return int(x) == 200
        except Exception:
            return False

    def run(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.masterlist_csv)

        required = {"url", "status", "rel_raw_txt_path"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"masterlist_csv missing columns: {missing}")

        df["status"] = pd.to_numeric(df["status"], errors="coerce").fillna(-1).astype(int)
        df["rel_raw_txt_path"] = df["rel_raw_txt_path"].astype(str)

        # Create output columns safely
        if "rel_clean_txt_path" not in df.columns:
            df["rel_clean_txt_path"] = ""
        else:
            df["rel_clean_txt_path"] = df["rel_clean_txt_path"].astype(str)

        if "title" not in df.columns:
            df["title"] = ""
        else:
            df["title"] = df["title"].astype(str)

        if "clean_error" not in df.columns:
            df["clean_error"] = ""
        else:
            df["clean_error"] = df["clean_error"].astype(str)

        if "clean_chars" not in df.columns:
            df["clean_chars"] = 0
        df["clean_chars"] = pd.to_numeric(df["clean_chars"], errors="coerce").fillna(0).astype(int)

        if "clean_lines" not in df.columns:
            df["clean_lines"] = 0
        df["clean_lines"] = pd.to_numeric(df["clean_lines"], errors="coerce").fillna(0).astype(int)

        if "clean_ok" not in df.columns:
            df["clean_ok"] = False
        df["clean_ok"] = df["clean_ok"].astype(bool)

        for i, row in df.iterrows():
            if not self._status_is_success(row["status"]):
                df.at[i, "clean_ok"] = False
                df.at[i, "clean_error"] = f"skip_status_{int(row['status'])}"
                continue

            rel_raw = str(row["rel_raw_txt_path"])
            res = self.postprocess_one(rel_raw)

            df.at[i, "rel_clean_txt_path"] = str(res["rel_clean_txt_path"])
            df.at[i, "title"] = str(res["title"])
            df.at[i, "clean_chars"] = int(res["clean_chars"])
            df.at[i, "clean_lines"] = int(res["clean_lines"])
            df.at[i, "clean_ok"] = bool(res["clean_ok"])
            df.at[i, "clean_error"] = str(res.get("error", ""))

        outp = Path(self.cfg.out_masterlist_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outp, index=False)

        n_ok = int(df["clean_ok"].sum())
        print(f"DONE. Clean OK: {n_ok} of {len(df)}.")
        return df






if __name__ == "__main__":
    RadiopaediaArticleURLExtractor(output_path = "/PATH/radio_urls.txt").run()



    cfg = RadiopaediaDownloadConfig(
        out_dir="/PATH/Articles_Jan_2026",
        url_list_path="/PATH/radio_urls.txt",
        timeout_sec=100,
        base_delay_sec=12.0,
        jitter_sec=6.0,
        masterlist_csv_path="/PATH/masterlist.csv",

    )
    downloader = RadiopaediaArticleDownloader(cfg)
    downloader.download_all()





    cfg = RadiopaediaPostprocessConfig(
        masterlist_csv="/PATH/masterlist.csv",
        raw_base_dir="/PATH/Articles_Jan_2026",
        out_dir_clean="/PATH/Articles_Processed_Jan_2026",
        out_masterlist_csv="/PATH/masterlist_with_clean.csv",
        min_clean_chars=800,
        overwrite=False,
        keep_references=True,
    )
    pp = RadiopaediaArticlePostprocessor(cfg)
    df_out = pp.run()
