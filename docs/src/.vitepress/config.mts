import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import mathjax3 from "markdown-it-mathjax3";
import footnote from "markdown-it-footnote";
import path from 'path'

function getBaseRepository(base: string): string {
  if (!base || base === '/') return '/';
  const parts = base.split('/').filter(Boolean);
  return parts.length > 0 ? `/${parts[0]}/` : '/';
}

const baseTemp = {
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
}

const navTemp = {
  nav: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
}

// ✅ add: sidebar placeholder
const sidebarTemp = {
  sidebar: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
}

// ✅ add: collapse helper
function collapseAllSidebarGroups(sidebar: any): any {
  const collapseInArray = (arr: any[]) =>
    arr.map((entry) => {
      if (!entry || typeof entry !== "object") return entry
      if (Array.isArray((entry as any).items)) {
        return {
          ...entry,
          collapsible: true,
          collapsed: true,
          items: collapseInArray((entry as any).items),
        }
      }
      return entry
    })

  if (Array.isArray(sidebar)) return collapseInArray(sidebar)

  if (sidebar && typeof sidebar === "object") {
    const out: Record<string, any> = {}
    for (const key of Object.keys(sidebar)) {
      const v = (sidebar as any)[key]
      out[key] = Array.isArray(v) ? collapseInArray(v) : v
    }
    return out
  }

  return sidebar
}

const nav = [
  ...navTemp.nav,
  { component: 'VersionPicker' }
]

export default defineConfig({
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  title: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  description: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  lastUpdated: true,
  cleanUrls: true,
  outDir: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  head: [
    ['link', { rel: 'icon', href: 'REPLACE_ME_DOCUMENTER_VITEPRESS_FAVICON' }],
    ['script', {src: `${getBaseRepository(baseTemp.base)}versions.js`}],
    ['script', {src: `${baseTemp.base}siteinfo.js`}]
  ],

  vite: {
    define: {
      __DEPLOY_ABSPATH__: JSON.stringify('REPLACE_ME_DOCUMENTER_VITEPRESS_DEPLOY_ABSPATH'),
    },
    resolve: {
      alias: { '@': path.resolve(__dirname, '../components') }
    },
    optimizeDeps: {
      exclude: [
        '@nolebase/vitepress-plugin-enhanced-readabilities/client',
        'vitepress',
        '@nolebase/ui',
      ],
    },
    ssr: {
      noExternal: [
        '@nolebase/vitepress-plugin-enhanced-readabilities',
        '@nolebase/ui',
      ],
    },
  },

  markdown: {
    math: true,
    config(md) {
      md.use(tabsMarkdownPlugin)
      md.use(mathjax3)
      md.use(footnote)
    },
    theme: { light: "github-light", dark: "github-dark" }
  },

  themeConfig: {
    outline: 'deep',
    logo: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    search: {
      provider: 'local',
      options: { detailedView: true }
    },
    nav,

    // ✅ collapse all groups by default
    sidebar: collapseAllSidebarGroups(sidebarTemp.sidebar as any),

    editLink: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    socialLinks: [
      { icon: 'github', link: 'https://github.com/sebapersson/PEtab.jl' }
    ],
    footer: {
      message: 'Made with <a href="https://luxdl.github.io/DocumenterVitepress.jl/dev/" target="_blank"><strong>DocumenterVitepress.jl</strong></a><br>',
      copyright: `© Copyright ${new Date().getUTCFullYear()}.`
    }
  }
})
