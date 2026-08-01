#!/usr/bin/env python
"""
organise_collections.py

Populate HuggingFace collections with ONLY the full-data LoRA adapters from the
rank sweep -- repos named  <namespace>/<Family>-<size>-LoRA-r<rank>  with NO
-f<fraction> suffix. The dataset-size ablation adapters (...-rN-f0.0625 etc.)
are excluded.

Two grouping modes:

  --by family            one collection per family (must already exist):
                             Qwentaur-LoRA, Llama-Centaur-LoRA,
                             Olmotaur-LoRA, Smoltaur-LoRA
  --by size  (default)   one collection per size, CREATED if missing:
                             Qwentaur-0.6B-LoRA ... Smoltaur-0.1B-LoRA

It discovers what actually exists on the Hub (list_models) rather than trusting
a hard-coded grid, and is idempotent: it only ever ADDS, skipping anything
already present, so re-running picks up newly uploaded adapters.

Auth: a write token for the `socius` namespace (huggingface-cli login / HF_TOKEN).

Usage
-----
    python organise_collections.py                 # dry run, by size
    python organise_collections.py --apply         # create per-size collections + add
    python organise_collections.py --by family --apply
"""

import argparse
import re

from huggingface_hub import (
    HfApi,
    add_collection_item,
    create_collection,
    get_collection,
    list_collections,
    update_collection_item,
)

# family key -> (repo-name family token, per-family collection title)
# Keys and tokens match ablation_naming.DISPLAY_NAME.
FAMILIES = {
    "qwen":   {"token": "Qwentaur",      "collection": "Qwentaur-LoRA"},
    "llama":  {"token": "Llama-Centaur", "collection": "Llama-Centaur-LoRA"},
    "olmo":   {"token": "Olmotaur",      "collection": "Olmotaur-LoRA"},
    "smollm": {"token": "Smoltaur",      "collection": "Smoltaur-LoRA"},
}

# full-data rank-sweep adapter: ...<Family>-<size>B-LoRA-r<rank>  (NO -f suffix)
# All families use B-suffixed sizes (Smoltaur's sub-B models are 0.1B/0.4B).
_FAM_ALT = "|".join(re.escape(v["token"]) for v in FAMILIES.values())
FULLDATA_RE = re.compile(
    rf"^(?P<ns>[^/]+)/(?P<fam>{_FAM_ALT})-"
    r"(?P<size>[0-9.]+B)-LoRA-r(?P<rank>\d+)$"
)

SIZE_VAL = lambda s: float(s.rstrip("B"))   # "0.6B" -> 0.6


def discover_fulldata_adapters(api, namespace):
    """Return {family_key: [(repo_id, size, rank), ...]} for full-data adapters."""
    buckets = {k: [] for k in FAMILIES}
    fam_token_to_key = {v["token"]: k for k, v in FAMILIES.items()}
    for m in api.list_models(author=namespace):
        match = FULLDATA_RE.match(m.id)
        if not match:
            continue
        key = fam_token_to_key[match.group("fam")]
        buckets[key].append((m.id, match.group("size"), int(match.group("rank"))))
    for key in buckets:
        buckets[key].sort(key=lambda t: (SIZE_VAL(t[1]), t[2]))   # size then rank
    return buckets


def find_collection_slug(title, owners):
    """Locate a collection by exact title across candidate owners. -> slug|None."""
    for owner in owners:
        try:
            for col in list_collections(owner=owner):
                if col.title == title:
                    return col.slug
        except Exception:
            continue
    return None


RANK_RE = re.compile(r"-LoRA-r(\d+)$")


def reorder_by_rank(slug, apply):
    """Sort a collection's model items ascending by LoRA rank (r4..r64).

    Positions are 0-indexed; setting each item to its target slot in increasing
    order converges to the sorted order. Idempotent -- a no-op if already sorted.
    """
    if slug is None:
        return
    items = [it for it in get_collection(slug).items if it.item_type == "model"]

    def rank_of(it):
        m = RANK_RE.search(it.item_id)
        return int(m.group(1)) if m else 1_000_000      # unparseable -> last

    target = sorted(items, key=rank_of)
    current_ids = [it.item_id for it in items]
    desired_ids = [it.item_id for it in target]
    if current_ids == desired_ids:
        print("    order: already sorted by rank")
        return
    print("    order: " + " -> ".join(
        ("r%s" % rank_of(it)) for it in target))
    if apply:
        for pos, it in enumerate(target):
            update_collection_item(collection_slug=slug,
                                   item_object_id=it.item_object_id, position=pos)


def add_items(slug, items, apply):
    """Add repos to a collection (no per-item note), skipping any already in it."""
    existing = set()
    if slug is not None:
        existing = {it.item_id for it in get_collection(slug).items
                    if it.item_type == "model"}
    to_add = [r for r in items if r[0] not in existing]
    for rid, _size, rank in to_add:
        print(f"      + {rid:<48} (rank {rank})")
        if apply and slug is not None:
            add_collection_item(collection_slug=slug, item_id=rid,
                                item_type="model", exists_ok=True)
    print(f"    -> add {len(to_add)}, skip {len(items) - len(to_add)} already present")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--namespace", default="socius",
                    help="Owner of the adapter repos AND of created collections.")
    ap.add_argument("--by", choices=["family", "size"], default="size",
                    help="Group into per-family or per-size collections (default: size).")
    ap.add_argument("--collection-owners", nargs="+", default=None,
                    help="Where to look for existing collections "
                         "(default: <namespace> then the logged-in user).")
    ap.add_argument("--family", choices=sorted(FAMILIES), default=None,
                    help="Only process one family (default: both).")
    ap.add_argument("--private", dest="private", action="store_true", default=None,
                    help="Create per-size collections private "
                         "(default: match the existing per-family collection).")
    ap.add_argument("--public", dest="private", action="store_false",
                    help="Create per-size collections public.")
    ap.add_argument("--reorder", action="store_true",
                    help="Also sort items within each collection ascending by rank.")
    ap.add_argument("--apply", action="store_true",
                    help="Actually create collections / add items (else dry run).")
    ap.add_argument("--token", default=None, help="HF write token override.")
    args = ap.parse_args()

    api = HfApi(token=args.token)
    me = api.whoami()
    owners = args.collection_owners or [args.namespace, me["name"]]
    fams = [args.family] if args.family else list(FAMILIES)
    buckets = discover_fulldata_adapters(api, args.namespace)

    print("=" * 74)
    print(f"  Organise collections by {args.by}  "
          f"(auth: {me['name']}, {'APPLY' if args.apply else 'DRY RUN'})")
    print("=" * 74)

    # ----------------------------------------------------------------- family
    if args.by == "family":
        for key in fams:
            title = FAMILIES[key]["collection"]
            print(f"\n[{FAMILIES[key]['token']}]  ->  \"{title}\"")
            slug = find_collection_slug(title, owners)
            if slug is None:
                print(f"  !! \"{title}\" not found under {owners}; create it first.")
                continue
            print(f"  slug: {slug}")
            add_items(slug, buckets[key], args.apply)
            if args.reorder:
                reorder_by_rank(slug, args.apply)
        print("\n" + ("Applied." if args.apply else "Dry run -- re-run with --apply."))
        return

    # ------------------------------------------------------------------- size
    # Visibility for newly created collections: match the existing per-family
    # collection unless overridden with --private/--public.
    visibility = args.private
    if visibility is None:
        ref = next((slug for fam in FAMILIES.values()
                    if (slug := find_collection_slug(fam["collection"], owners))),
                   None)
        visibility = bool(getattr(get_collection(ref), "private", False)) if ref else False
    print(f"  new collections -> namespace={args.namespace}, "
          f"private={visibility}\n")

    for key in fams:
        token = FAMILIES[key]["token"]
        by_size = {}
        for rid, size, rank in buckets[key]:
            by_size.setdefault(size, []).append((rid, size, rank))

        for size in sorted(by_size, key=SIZE_VAL):
            title = f"{token}-{size}-LoRA"
            print(f"[{title}]  ({len(by_size[size])} adapter(s))")
            slug = find_collection_slug(title, owners)
            if slug is None:
                print(f"  (will create, private={visibility})"
                      if not args.apply else "")
                if args.apply:
                    col = create_collection(title=title, namespace=args.namespace,
                                            private=visibility, exists_ok=True)
                    slug = col.slug
                    print(f"  created: {slug}")
            else:
                print(f"  exists: {slug}")
            add_items(slug, by_size[size], args.apply)
            if args.reorder:
                reorder_by_rank(slug, args.apply)
            print()

    print("Applied." if args.apply else "Dry run -- re-run with --apply.")


if __name__ == "__main__":
    main()
