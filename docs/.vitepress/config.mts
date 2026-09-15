import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import llmstxt from 'vitepress-plugin-llms'

// https://vitepress.dev/reference/site-config
export default withMermaid(
  defineConfig({
    title: 'Tiny-LLM',
    description: '面向聚焦型 Transformer 工作负载的 CUDA 原生 C++ 推理引擎',

    // Base URL for GitHub Pages deployment
    base: '/tiny-llm/',

    // Clean URLs without .html extension
    cleanUrls: true,

    // Last updated timestamp
    lastUpdated: true,

    // Head configuration
    head: [
      ['meta', { name: 'theme-color', content: '#00D4AA' }],
      ['meta', { name: 'og:type', content: 'website' }],
      ['meta', { name: 'og:title', content: 'Tiny-LLM | CUDA 原生推理引擎' }],
      ['meta', { name: 'og:description', content: '面向聚焦型 Transformer 工作负载的 CUDA 原生 C++ 推理引擎' }],
      ['link', { rel: 'icon', href: '/tiny-llm/favicon.svg' }],
      ['link', { rel: 'apple-touch-icon', href: '/tiny-llm/apple-touch-icon.png' }],
      // Google Fonts
      ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
      ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
      ['link', { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap' }],
    ],

    // Markdown configuration
    markdown: {
      theme: {
        light: 'github-light',
        dark: 'github-dark',
      },
      lineNumbers: true,
    },

    // 单语站点：root 即中文，无需 i18n locale 切换
    lang: 'zh-CN',

    // Theme configuration
    themeConfig: {
      logo: '/logo.svg',
      siteTitle: 'Tiny-LLM',

      nav: [
        { text: '架构', link: '/architecture/' },
        { text: '性能', link: '/performance/' },
        { text: 'API', link: '/api/' },
        { text: '指南', link: '/guide/getting-started' },
      ],

      sidebar: {
        '/architecture/': [
          {
            text: '架构',
            items: [
              { text: '概述', link: '/architecture/' },
              { text: '推理引擎', link: '/architecture/inference-engine' },
              { text: 'W8A16 量化', link: '/architecture/quantization' },
              { text: 'KV 缓存设计', link: '/architecture/kv-cache' },
              { text: 'CUDA 内核', link: '/architecture/cuda-kernels' },
              { text: '内存模型', link: '/architecture/memory-model' },
              { text: 'Split-KV Decode Attention（设计）', link: '/architecture/decode-attention-splitkv-design' },
            ],
          },
        ],
        '/performance/': [
          {
            text: '性能',
            items: [
              { text: '概述', link: '/performance/' },
              { text: '基准测试', link: '/performance/benchmarks' },
              { text: '对比方法论', link: '/performance/benchmark-methodology' },
              { text: 'CUDA Graphs', link: '/performance/cuda-graphs' },
              { text: '分析指南', link: '/performance/profiling-guide' },
              { text: '分析概览', link: '/performance/profiling' },
              { text: '优化', link: '/performance/optimization' },
            ],
          },
        ],
        '/api/': [
          {
            text: 'API 参考',
            items: [
              { text: '概述', link: '/api/' },
              { text: 'InferenceEngine', link: '/api/inference-engine' },
              { text: 'ModelConfig', link: '/api/model-config' },
              { text: 'Result<T>', link: '/api/result' },
              { text: 'KVCacheManager', link: '/api/kv-cache' },
              { text: '参考资料', link: '/api/references' },
            ],
          },
        ],
        '/guide/': [
          {
            text: '指南',
            items: [
              { text: '入门指南', link: '/guide/getting-started' },
              { text: '安装', link: '/guide/installation' },
              { text: '快速开始', link: '/guide/quickstart' },
              { text: '配置', link: '/guide/configuration' },
              { text: '量化', link: '/guide/quantization' },
              { text: '故障排除', link: '/guide/troubleshooting' },
            ],
          },
        ],
        '/contributing/': [
          {
            text: '贡献',
            items: [
              { text: '开发者指南', link: '/contributing/' },
            ],
          },
        ],
      },

      socialLinks: [
        { icon: 'github', link: 'https://github.com/open-infra-ai/tiny-llm' },
      ],

      editLink: {
        pattern: 'https://github.com/open-infra-ai/tiny-llm/edit/master/docs/:path',
        text: '在 GitHub 上编辑此页',
      },

      footer: {
        message: '基于 MIT 许可证发布',
        copyright: '版权所有 © 2024至今 Tiny-LLM 贡献者',
      },

      docFooter: {
        prev: '上一页',
        next: '下一页',
      },

      outline: {
        label: '页面导航',
      },

      lastUpdated: {
        text: '最后更新于',
        formatOptions: {
          dateStyle: 'medium',
          timeStyle: 'short',
        },
      },

      returnToTopLabel: '返回顶部',
      sidebarMenuLabel: '菜单',
      darkModeSwitchLabel: '主题',
      lightModeSwitchTitle: '切换到浅色模式',
      darkModeSwitchTitle: '切换到深色模式',

      search: {
        provider: 'local',
        options: {
          translations: {
            button: {
              buttonText: '搜索',
              buttonAriaLabel: '搜索',
            },
            modal: {
              noResultsText: '无搜索结果',
              resetButtonTitle: '重置搜索',
              footer: {
                selectText: '选择',
                navigateText: '导航',
                closeText: '关闭',
              },
            },
          },
        },
      },
    },

    // Mermaid configuration
    mermaid: {
      theme: 'base',
      themeVariables: {
        primaryColor: '#00D4AA',
        primaryTextColor: '#fff',
        primaryBorderColor: '#00C49A',
        lineColor: '#76B900',
        secondaryColor: '#F59E0B',
        tertiaryColor: '#8B5CF6',
      },
    },

    // Build options
    build: {
      chunkSizeWarningLimit: 1500,
    },

    // Vite config
    vite: {
      build: {
        minify: 'terser',
        chunkSizeWarningLimit: 1500,
      },
    },
  })
)

// Plugin: llmstxt for LLM-friendly documentation
llmstxt({
  domain: 'https://open-infra-ai.github.io/tiny-llm',
  title: 'Tiny-LLM 文档',
  description: '面向聚焦型 Transformer 工作负载的 CUDA 原生 C++ 推理引擎',
  sections: {
    'Architecture': {
      title: '架构',
      description: '系统架构与设计',
    },
    'Performance': {
      title: '性能',
      description: '基准测试与优化',
    },
    'API': {
      title: 'API 参考',
      description: '完整 API 文档',
    },
    'Guide': {
      title: '指南',
      description: '入门与使用指南',
    },
  },
})
