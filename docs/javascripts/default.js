function parseURL(e, t) {
  let r = "",
    n = "";
  return (
    0 === t
      ? ((r = e.href), (n = e.title))
      : 1 === type_url && ((r = e.src), (n = e.alt)),
    (r = (r = r.match(/index$/) ? r.replace(/index$/, "") : r).includes("%5C")
      ? r.replace(/%5C/g, "/")
      : r).match(/\.md\/$/) && (r = r.replace(/\.md\/$/, "/")),
    (r = decodeURI(r)),
    0 === t
      ? ((e.href = r),
        0 === (e.title = n).length && ((n = e.innerText), (e.title = n)))
      : 1 === t && ((e.src = r), (e.alt = n)),
    { title: n, ref: r, url: e }
  );
}
function checkIfInternalLinksExists(t, r, n, i) {
  let l = n.href
    .replace(n.host, "")
    .replace(/http(s)?:(\/){1,3}/gi, "")
    .replace(/^\//, "");
  l = 0 === l.trim().length ? "./" : decodeURI(l).toLowerCase();
  var e = document.querySelector('meta[name="site_url"]').content,
    o =
      e
        .split("/")
        .filter((e) => 0 < e.length)
        .pop() + "/";
  return (
    (l = l.replace(o.replace(/^\//, ""), "")),
    i.includes(l.replace(/\/$/, "")) ||
      "./" === l ||
      fetch(e + "/search/all_files.json")
        .then((e) => e.json())
        .then((e) => {
          if (
            (e.forEach((e) => {
              decodeURI(e.url).toLowerCase() === l &&
                i.push(l.replace(/\/$/, ""));
            }),
            !(i = [...new Set(i)]).includes(l.replace(/\/$/, "")) && "./" !== l)
          ) {
            e = document.createElement("div");
            (e.innerHTML = r),
              e.classList.add("not_found"),
              e.setAttribute("href", t);
            try {
              n.parentNode.replaceChild(e, n);
            } catch (e) {
              console.log(e);
            }
          }
        })
        .catch((e) => {
          console.log(e);
        }),
    i
  );
}
let history = [];
var p_search = /\.{2}\//gi,
  ht = document.querySelectorAll("a:not(img)");
for (const v of ht)
  0 < !v.getElementsByTagName("img").length &&
    0 < !v.getElementsByTagName("svg").length &&
    !v.href.includes("#") &&
    v.hostname === window.location.hostname &&
    ((link = parseURL(v, 0)),
    (history = checkIfInternalLinksExists(link.ref, link.title, v, history)));
var img,
  link,
  partReg,
  size,
  p_img = /\.+\\/gi;
for (const w of (img = document.querySelectorAll("img")))
  w.hostname === window.location.hostname &&
    ((link = parseURL(w, 1)),
    (history = checkIfInternalLinksExists(link.ref, link.title, w, history)));
const mkDocsChirpyTranslator = { default: "light", slate: "dark" },
  mkDocs = document.querySelector("[data-md-color-scheme]"),
  chirpy = document.querySelector("[data-chirpy-theme]");
if (chirpy) {
  "default" === mkDocs.getAttribute("data-md-color-scheme") &&
    chirpy.setAttribute("data-chirpy-theme", "light");
  const x = new MutationObserver((e) => {
    e.forEach((e) => {
      "attributes" === e.type &&
        chirpy.setAttribute(
          "data-chirpy-theme",
          mkDocsChirpyTranslator[mkDocs.dataset.mdColorScheme]
        );
    });
  });
  x.observe(mkDocs, {
    attributes: !0,
    attributeFilter: ["data-md-color-scheme"],
  });
}
const header_links = document.querySelectorAll('a[href*="#"]');
if (header_links)
  for (var i = 0; i < header_links.length; i++) {
    const A = header_links[i].getAttribute("href").replace("^.*#", "");
    let e = A.replace(/\s/g, "-");
    (e = A.normalize("NFD").replace(/[\u0300-\u036f]/g, "")),
      header_links[i].setAttribute(
        "href",
        header_links[i].getAttribute("href").replace(A, e)
      );
  }
function getHeightWidth(e) {
  var t = new RegExp("\\d+x\\d+"),
    r = new RegExp("\\d+");
  return e.match(t)
    ? [parseInt(e.split("x")[0]), parseInt(e.split("x")[1])]
    : e.match(r)
    ? [parseInt(e.match(r)[0]), 0]
    : [0, 0];
}
const p_img = /\.+\\/gi,
  img = document.querySelectorAll("img");
for (const i of img) {
  const H = new RegExp("\\|"),
    I = i.alt;
  if (I.match(H)) {
    const J = I.split("|");
    for (const K of J)
      K.match(new RegExp("\\d+", "g")) &&
        ((size = getHeightWidth(K)),
        (i.width = 0 < size[0] ? size[0] : i.width),
        (i.height = 0 < size[1] ? size[1] : i.height),
        (partReg = new RegExp("\\" + K)),
        (i.alt = I.replace(partReg, "")));
  } else
    I.match(/\d+/g) &&
      ((size = getHeightWidth(I)),
      (i.width = 0 < size[0] ? size[0] : i.width),
      (i.height = 0 < size[1] ? size[1] : i.height),
      (i.alt = I.replace(p_img, "")));
}
const article = document.querySelectorAll(
    "article.md-content__inner.md-typeset > *:not(.highlight)"
  ),
  embed_id_regex = /\^\w+\s*$/gi;
for (const L of article) {
  const M = L.innerText.match(embed_id_regex);
  M && (L.innerHTML = L.innerText.replace(M, ""));
}
document.innerText = article;
const cite = document.querySelectorAll(".citation");
if (cite)
  for (i = 0; i < cite.length; i++) {
    const N = cite[i].innerHTML.match(/!?(\[{2}|\[).*(\]{2}|\))/gi);
    if (N) {
      for (var j = 0; j < N.length; j++)
        cite[i].innerHTML = cite[i].innerHTML.replace(N[j], "");
      cite[i].innerText.trim().length < 2 && (cite[i].style.display = "none");
    }
  }
window.onload = function () {
  var e = document.querySelector("iframe");
  if (e) {
    const t = [];
    document.querySelectorAll("link").forEach((e) => {
      e.href.endsWith(".css") && t.push(e.href);
    });
    const r = e.contentDocument || e.contentWindow.document;
    t.forEach((e) => {
      var t = document.createElement("link");
      (t.rel = "stylesheet"),
        (t.href = e),
        (t.type = "text/css"),
        r.head.appendChild(t);
    });
    var e = document.querySelector("[data-md-color-scheme]");
    "default" === e.getAttribute("data-md-color-scheme")
      ? r.body.setAttribute("class", "light")
      : (r.body.setAttribute("class", "dark"),
        (e = getComputedStyle(e).getPropertyValue("--md-default-bg-color")),
        r.body.style.setProperty("--md-default-bg-color", e)),
      r.body.classList.add("graph-view");
  }
};
const paletteSwitcher1 = document.getElementById("__palette_1"),
  paletteSwitcher2 = document.getElementById("__palette_2"),
  isMermaidPage = document.querySelector(".mermaid"),
  blogURL =
    (isMermaidPage &&
      (paletteSwitcher1.addEventListener("change", function () {
        location.reload();
      }),
      paletteSwitcher2.addEventListener("change", function () {
        location.reload();
      })),
    document.querySelector('meta[name="site_url"]')
      ? document.querySelector('meta[name="site_url"]').content
      : location.origin);
let position = ["top", "right", "bottom", "left"];
function brokenImage(e) {
  var t = e?.querySelectorAll("img");
  if (t)
    for (let e = 0; e < t.length; e++) {
      var r = t[e];
      (r.src = decodeURI(decodeURI(r.src))),
        (r.src = r.src.replace(location.origin, blogURL));
    }
  return e;
}
function cleanText(e) {
  return (e.innerText = e.innerText.replaceAll("↩", "").replaceAll("¶", "")), e;
}
function calculateHeight(e) {
  (e = e ? e.innerText || e : ""), (e = Math.floor(e.split(" ").length / 100));
  return e < 2 ? "auto" : 5 <= e ? "20rem" : e + "rem";
}
try {
  tippy(`.md-content a[href^="${blogURL}"], a.footnote-ref, a[href^="./"]`, {
    content: "",
    allowHTML: !0,
    animation: "scale-subtle",
    theme: "translucent",
    followCursor: !0,
    arrow: !1,
    touch: "hold",
    inlinePositioning: !0,
    placement: position[Math.floor(Math.random() * position.length - 1)],
    onShow(l) {
      fetch(l.reference.href)
        .then((e) => e.text())
        .then((e) => {
          return new DOMParser().parseFromString(e, "text/html");
        })
        .then((i) => {
          return (
            i.querySelectorAll("h1, h2, h3, h4, h5, h6").forEach(function (t) {
              var r =
                t.id ||
                t.innerText.split("\n")[0].toLowerCase().replaceAll(" ", "-");
              if (0 < r.length) {
                var n = i.createElement("div");
                n.classList.add(r);
                let e = t.nextElementSibling;
                for (; e && !e.matches("h1, h2, h3, h4, h5, h6"); )
                  n.appendChild(e), (e = e.nextElementSibling);
                t.parentNode.insertBefore(n, t.nextSibling);
              }
            }),
            i
          );
        })
        .then((r) => {
          if (location.href.replace(location.hash, "") === l.reference.href)
            l.hide(), l.destroy();
          else {
            let e = r.querySelector("article");
            var n = r.querySelector("h1"),
              n =
                (n &&
                  "Index" === n.innerText &&
                  ((i = decodeURI(r.querySelector('link[rel="canonical"]').href)
                    .split("/")
                    .filter((e) => e)
                    .pop()),
                  (n.innerText = i)),
                (e = brokenImage(e)),
                document.querySelector('[id^="tippy"]')),
              i =
                (n && n.classList.add("tippy"),
                l.reference.href.replace(/.*#/, "#"));
            let t = e;
            i.startsWith("#")
              ? ((e = r.querySelector(
                  `[id="${i.replace("#", "")}"]`
                )).tagName.includes("H")
                  ? ((n = r.createElement("article")).classList.add(
                      "md-content__inner",
                      "md-typeset"
                    ),
                    n.appendChild(r.querySelector("div." + i.replace("#", ""))),
                    (t = n),
                    (e = t))
                  : (t =
                      0 === e.innerText.replace(i).length
                        ? (e = r.querySelector("div.citation"))
                        : cleanText(e).innerText),
                (l.popper.style.height = "auto"))
              : (l.popper.style.height = calculateHeight(e)),
              (l.popper.placement =
                position[Math.floor(Math.random() * position.length)]),
              0 < e.innerText.length
                ? (l.setContent(t),
                  (l.popper.style.height = calculateHeight(t)))
                : ((e = r.querySelector("article")),
                  l.reference.href.replace(/.*#/, "#"),
                  (l.popper.style.height = calculateHeight(e)));
          }
        })
        .catch((e) => {
          console.log(e), l.hide(), l.destroy();
        });
    },
  });
} catch {
  console.log("tippy error, ignore it");
}
