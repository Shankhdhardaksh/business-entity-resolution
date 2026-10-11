"""Stage 1 — preprocessing. Run once: python preprocess.py <raw_dir> <out_dir>"""
import sys, pathlib, polars as pl
from anyascii import anyascii

# ---------- dictionaries ----------
LEGAL = {  # variant -> canonical legal token
 **dict.fromkeys(["private","pvt","pvte","prvt","praivet","piraivet","praibhet","praivrr","pra"],"pvt"),
 **dict.fromkeys(["limited","ltd","limitet","limirrd","li","ltda"],"ltd"),
 **dict.fromkeys(["incorporated","inc"],"inc"), **dict.fromkeys(["corporation","corp"],"corp"),
 **dict.fromkeys(["company","co"],"co"), **dict.fromkeys(["llc"],"llc"), **dict.fromkeys(["llp","elelpi"],"llp"),
 **dict.fromkeys(["lp"],"lp"), **dict.fromkeys(["plc"],"plc"), **dict.fromkeys(["pllc"],"pllc"), **dict.fromkeys(["pc"],"pc"),
 **{k: k for k in ["sarl","sas","sasu","eurl","sci","sa","snc","selarl","scop"]},
 "etablissements":"ets","ets":"ets",
}
STOP = {"the","of","and","india","france","ets","de","du","des","la","le","les","l","d","et"}
US = dict(alabama="AL",alaska="AK",arizona="AZ",arkansas="AR",california="CA",colorado="CO",connecticut="CT",delaware="DE",
 florida="FL",georgia="GA",hawaii="HI",idaho="ID",illinois="IL",indiana="IN",iowa="IA",kansas="KS",kentucky="KY",louisiana="LA",
 maine="ME",maryland="MD",massachusetts="MA",michigan="MI",minnesota="MN",mississippi="MS",missouri="MO",montana="MT",nebraska="NE",
 nevada="NV",ohio="OH",oklahoma="OK",oregon="OR",pennsylvania="PA",tennessee="TN",texas="TX",utah="UT",vermont="VT",virginia="VA",
 washington="WA",wisconsin="WI",wyoming="WY",**{"new hampshire":"NH","new jersey":"NJ","new mexico":"NM","new york":"NY",
 "north carolina":"NC","north dakota":"ND","rhode island":"RI","south carolina":"SC","south dakota":"SD","west virginia":"WV",
 "district of columbia":"DC"})
IN = {"maharashtra":"MH","delhi":"DL","uttar pradesh":"UP","karnataka":"KA","tamil nadu":"TN","west bengal":"WB","gujarat":"GJ",
 "telangana":"TG","haryana":"HR","rajasthan":"RJ","kerala":"KL","andhra pradesh":"AP","bihar":"BR","madhya pradesh":"MP",
 "punjab":"PB","orissa":"OD","odisha":"OD","assam":"AS","jharkhand":"JH","chhattisgarh":"CG","goa":"GA","uttarakhand":"UK",
 "himachal pradesh":"HP","jammu and kashmir":"JK","chandigarh":"CH","puducherry":"PY","pondicherry":"PY","ts":"TG",
 # regional-script state names seen in the data (the only Indic text in addresses)
 "महाराष्ट्र":"MH","दिल्ली":"DL","उत्तर प्रदेश":"UP","ಕರ್ನಾಟಕ":"KA","தமிழ்நாடு":"TN","ગુજરાત":"GJ","পশ্চিমবঙ্গ":"WB",
 "తెలంగాణ":"TG","हरियाणा":"HR","राजस्थान":"RJ","കേരളം":"KL","बिहार":"BR","मध्य प्रदेश":"MP","ఆంధ్రప్రదేశ్":"AP","ਪੰਜਾਬ":"PB","ଓଡ଼ିଶା":"OD"}
FR = {"hauts-de-france":"HDF","nord":"HDF","pas-de-calais":"HDF","nouvelle-aquitaine":"NAQ","gironde":"NAQ",
      "pays de la loire":"PDL","loire-atlantique":"PDL"}
STATE = {**US, **IN, **FR}
ALLCODES = sorted(set(STATE.values()))
CODES = {"US": sorted(set(US.values())), "India": sorted(set(IN.values())), "France": sorted(set(FR.values()))}
FR_CITY = {**dict.fromkeys(["bordeaux","pessac","merignac","teste de buch","lege cap ferret"],"NAQ"),
           **dict.fromkeys(["lille","tourcoing","roubaix","dunkerque","calais"],"HDF"),
           **dict.fromkeys(["nantes","st nazaire","pornic","baule escoublac","st herblain"],"PDL")}  # S3 often drops the region
ADDR = {  # token -> canonical
 "street":"st","str":"st","drive":"dr","avenue":"ave","av":"ave","road":"rd","court":"ct","lane":"ln","boulevard":"blvd","bd":"blvd",
 "place":"pl","circle":"cir","highway":"hwy","parkway":"pkwy","trail":"trl","square":"sq","terrace":"ter","way":"wy",
 "north":"n","south":"s","east":"e","west":"w","apartment":"apt","suite":"ste","floor":"fl","fl":"fl","r":"rue",
 "impasse":"imp","allee":"allee","saint":"st","sainte":"ste",
 "bengaluru":"bangalore","bombay":"mumbai","gurugram":"gurgaon","poona":"pune","calcutta":"kolkata","madras":"chennai",
}
ADDR_DROP = {"no","h","hno","door","nº","unit","cdp","township","po","box","near","opp","c","o","null","plot","vill","post","dist","at"}
NULLS = ["null","n/a","<null>","none","nan","na"]
PHON = [("tio","sio"),("ph","f"),("jh","j"),("bh","b"),("th","t"),("sh","s"),("kh","k"),("gh","g"),("dh","d"),("ch","c"),("ck","k"),("q","k"),
        ("c","k"),("x","ks"),("z","s"),("j","k"),("w","v"),("d","t"),("g","k"),("[mn]","n")]

tr = lambda e: e.map_elements(anyascii, return_dtype=pl.String, skip_nulls=True)
toks = lambda e: e.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))

def leet(e):  # digit adjacent to a letter -> letter (0->o, 1->l, 3->e, 4->a, 5->s, 8->b)
    for d, l in zip("013458", "oleasb"):
        e = e.str.replace_all(rf"([a-z]){d}", f"${{1}}{l}").str.replace_all(rf"{d}([a-z])", f"{l}${{1}}")
    return e

def clean_name(n):
    n = n.str.strip_chars().str.replace(r"\s*\|.*$", "")                       # "X | www.x.com" -> X
    n = pl.coalesce(n.str.extract(r"(?i)\s(?:aka|a/k/a|dba:?|d/b/a|fka|f/k/a|formerly known as|trading as|t/a)\s+(.+)$"), n)
    n = tr(n).str.to_lowercase()
    n = n.str.replace(r"^www\.", "").str.replace(r"\.(com|in|net|org|co|fr|biz|io|us)$", "")   # website -> stem
    n = leet(n.str.replace_all("'", "").str.replace_all("&", " and ").str.replace_all(r"\b([a-z])\.", "${1}"))       # l.l.c. -> llc
    return n.str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()

def indic_fix(n):  # anyascii artifacts: anusvara 'm' before consonant -> 'n' (tredimg->treding, imtrnesnl->intrnesnl)
    return n.str.replace_all(r"m([cdfghjklnqrstvxz])", "n${1}")

def phon(e):  # consonant skeleton: survives Indic transliteration + typos
    e = e.str.replace_all(r"[^a-z ]", "")
    for a, b in PHON: e = e.str.replace_all(a, b)
    e = e.str.replace_all(r"\B[aeiouy]", "")
    for c in "bfjklnprstv": e = e.str.replace_all(c + "+", c)  # collapse doubles (no backrefs in Rust regex)
    return e.str.replace_all(r"\s+", " ").str.strip_chars()

def clean_addr(a):
    a = a.str.split(",").list.eval(pl.element().str.strip_chars().str.to_lowercase()
          .replace(STATE).filter(~pl.element().is_in(NULLS) & (pl.element() != ""))).list.join(", ")  # NULL parts out, states -> code
    return tr(a).str.to_lowercase().str.replace_all(r"#+", " ").str.replace_all(r"\b0+(\d)", "${1}")  # ##12 -> 12, 002704 -> 2704

def prep(lf: pl.LazyFrame) -> pl.LazyFrame:
    s, E = pl.col, pl.element
    isst = E().str.to_uppercase().is_in(ALLCODES)
    return (lf.with_columns(
        name_kind=pl.when(s("business_name").str.contains(r"^\s*[#@]\S+$")).then(pl.lit("handle"))
                  .when(s("business_name").str.contains(r"(?i)\s(aka|dba:?|d/b/a|fka|formerly known as|trading as|t/a)\s")).then(pl.lit("alias"))
                  .when(s("business_name").str.contains(r"(?i)(^www\.|\.(com|in|net|org|co|fr|biz|io|us)$)")).then(pl.lit("web"))
                  .otherwise(pl.lit("plain")),
        is_indic=s("business_name").str.contains(r"[\u0900-\u0DFF]"),
        n=clean_name(s("business_name")),
        parts=clean_addr(s("business_address").fill_null("")).str.split(", "))
     .with_columns(n=pl.when(s("is_indic")).then(indic_fix(s("n"))).otherwise(s("n")),
                   state=pl.coalesce(pl.when(s("country") == k).then(s("parts").list.eval(E().filter(E().str.to_uppercase().is_in(v))).list.first().str.to_uppercase()) for k, v in CODES.items()),
                   at=toks(s("parts").list.eval(E().filter(~isst)).list.join(" ").str.replace_all(r"[^a-z0-9]+", " "))
                      .list.eval(E().replace(ADDR).filter(~E().is_in(list(ADDR_DROP)))))
     .with_columns(nt=toks(s("n")).list.eval(E().replace(LEGAL)))
     .with_columns(nt=s("nt").list.eval(E().filter(E().ne_missing(E().shift(1)))))            # "pvt pvt" -> "pvt"
     .with_columns(core=s("nt").list.eval(E().filter(~E().is_in(list(set(LEGAL.values()) | STOP)))))
     .with_columns(
        name_clean=s("nt").list.join(" "),
        name_core=pl.when(s("core").list.len() > 0).then(s("core")).otherwise(s("nt")).list.join(" "),
        addr_clean=s("at").list.join(" "),
        addr_nums=s("at").list.eval(E().filter(E().str.contains(r"^\d+$"))).list.unique().list.sort().list.join(" "))
     .with_columns(name_phon=phon(s("name_core")),
                   state=pl.coalesce("state", pl.when(s("country") == "France").then(
                       s("addr_clean").str.extract("(" + "|".join(FR_CITY) + ")").replace(FR_CITY))))
     .select("entity_id", "name_clean", "name_core", "name_phon", "addr_clean", "addr_nums", "state", "country",
             "name_kind", "is_indic", addr_empty=s("addr_clean") == ""))

def run(src: str, dst: str):
    prep(pl.scan_csv(src, separator="\t", quote_char=None, infer_schema=False)).sink_parquet(dst)

if __name__ == "__main__":
    raw, out = map(pathlib.Path, sys.argv[1:3]); out.mkdir(parents=True, exist_ok=True)
    for f in sorted(raw.glob("*source*.tsv")):
        print("->", f.name); run(str(f), str(out / (f.stem + ".parquet")))
