"use client";

import { useEffect, useRef } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/", label: "Predizioni" },
  { href: "/kelly", label: "Portafoglio Kelly" },
  { href: "/modello", label: "Metodologia" },
];

export default function SiteHeader() {
  const pathname = usePathname();
  const headerRef = useRef<HTMLElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLUListElement>(null);
  const lastCollapsedWidthRef = useRef<number>(0);

  // Caricamento condizionale del polyfill popover solo su browser che non lo supportano
  useEffect(() => {
    if (typeof HTMLElement !== "undefined" && !Object.hasOwn(HTMLElement.prototype, "popover")) {
      const s = document.createElement("script");
      s.type = "module";
      s.src = "https://unpkg.com/@oddbird/popover-polyfill@latest/dist/popover.min.js";
      document.head.appendChild(s);
    }
  }, []);

  useEffect(() => {
    const header = headerRef.current;
    const inner = innerRef.current;
    const menu = menuRef.current;
    if (!header || !inner || !menu) return;

    const measure = () => {
      const currentWidth = header.getBoundingClientRect().width;
      const isCurrentlyWide = header.classList.contains("is-wide");
      if (!isCurrentlyWide && currentWidth <= lastCollapsedWidthRef.current) {
        return;
      }

      let wasOpen = false;
      try {
        wasOpen = menu.matches(":popover-open") || menu.classList.contains(":popover-open");
      } catch {
        wasOpen = menu.classList.contains(":popover-open");
      }
      if (wasOpen && typeof menu.hidePopover === "function") {
        try {
          menu.hidePopover();
        } catch {
          // ignore
        }
      }

      header.classList.add("is-measuring");
      header.classList.add("is-wide");
      menu.removeAttribute("popover");

      const fits = inner.scrollWidth <= inner.clientWidth + 1;

      if (fits) {
        lastCollapsedWidthRef.current = 0;
      } else {
        header.classList.remove("is-wide");
        menu.setAttribute("popover", "auto");
        lastCollapsedWidthRef.current = currentWidth;
        if (wasOpen && typeof menu.showPopover === "function") {
          try {
            menu.showPopover();
          } catch {
            // ignore
          }
        }
      }

      requestAnimationFrame(() => {
        header.classList.remove("is-measuring");
      });
    };

    lastCollapsedWidthRef.current = 0;
    measure();
    const ro = new ResizeObserver(() => measure());
    ro.observe(header);
    ro.observe(menu);

    if (document.fonts?.ready) {
      document.fonts.ready.then(() => {
        lastCollapsedWidthRef.current = 0;
        measure();
      });
    }

    return () => {
      ro.disconnect();
    };
  }, [pathname]);

  const handleLinkClick = () => {
    const header = headerRef.current;
    const menu = menuRef.current;
    if (header && !header.classList.contains("is-wide") && menu && typeof menu.hidePopover === "function") {
      try {
        menu.hidePopover();
      } catch {
        // ignore
      }
    }
  };

  return (
    <header ref={headerRef} className="site-header">
      <div ref={innerRef} className="header-inner">
        <a className="skip-link" href="#main-content">
          Vai al contenuto
        </a>

        <Link href="/" className="site-brand" aria-label="CatBoost home">
          <svg
            className="brand-mark"
            width="20"
            height="20"
            viewBox="0 0 20 20"
            fill="none"
            aria-hidden="true"
          >
            <rect x="2" y="10" width="4" height="8" rx="1" fill="#3fc07a" />
            <rect x="8" y="6" width="4" height="12" rx="1" fill="#4a8ad6" />
            <rect x="14" y="12" width="4" height="6" rx="1" fill="#5899e2" />
            <path d="M2 5.5L8 3.5L14 6.5L18 4" stroke="#4a8ad6" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className="brand-text">
            <span className="brand-name">CatBoost</span>
            <span className="brand-sub">Pi-Rating · Dixon-Coles</span>
          </span>
        </Link>

        <nav className="site-nav" aria-label="Navigazione principale">
          <button
            className="menu-button"
            type="button"
            popoverTarget="site-menu"
            popoverTargetAction="toggle"
            aria-controls="site-menu"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path d="M2.5 4.5h11M2.5 8h11M2.5 11.5h11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            <span>Menu</span>
          </button>

          <ul ref={menuRef} id="site-menu" className="site-menu" popover="auto">
            {NAV_ITEMS.map((item) => {
              const isCurrent = item.href === "/" ? pathname === "/" : pathname?.startsWith(item.href);
              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    aria-current={isCurrent ? "page" : undefined}
                    onClick={handleLinkClick}
                  >
                    {item.label}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>
      </div>
    </header>
  );
}
