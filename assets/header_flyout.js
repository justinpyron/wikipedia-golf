/* Close About/Settings flyout on outside click (no full-page modal). */
(function () {
  function closeIfOutside(e) {
    var anchor = document.getElementById("header-flyout-anchor");
    if (!anchor || !anchor.classList.contains("is-open")) {
      return;
    }
    if (anchor.contains(e.target)) {
      return;
    }
    var dc = window.dash_clientside;
    if (dc && typeof dc.set_props === "function") {
      dc.set_props("header-flyout-store", { data: null });
    }
  }

  function bind() {
    document.addEventListener("mousedown", closeIfOutside, false);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind);
  } else {
    bind();
  }
})();
