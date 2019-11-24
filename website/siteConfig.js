/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// See https://docusaurus.io/docs/site-config for all the possible
// site configuration options.

// List of projects/orgs using your project for the users page.
const users = [
  {
    caption: 'User1',
    // You will need to prepend the image path with your baseUrl
    // if it is not '/', like: '/test-site/img/image.jpg'.
    image: '/img/undraw_open_source.svg',
    infoLink: 'https://www.facebook.com',
    pinned: true
  }
];

const siteConfig = {
  title: 'AquilaDB', // Title for your website.
  tagline: 'Inevitable muscle memory for your Machine Learning Applications',
  url: 'https://aquiladb.xyz', // Your website URL
  baseUrl: '/',
  projectName: 'aquiladb-documentation',
  organizationName: 'a-mma',
  // For no header links in the top nav bar -> headerLinks: [],
  headerLinks: [
    { languages: false },
    { doc: 'introduction', label: 'Docs' },
    { doc: 'api-reference', label: 'API' },
    { page: 'help', label: 'Help' },
    { href: 'https://medium.com/a-mma', label: 'Blog' },
    { href: 'https://github.com/a-mma/AquilaDB', label: 'GitHub' }
  ],

  /* path to images for header/footer */
  headerIcon: 'img/favicon_io/favicon.ico',
  footerIcon: 'img/favicon_io/favicon.ico',
  favicon: 'img/favicon_io/favicon.ico',

  /* Colors for website */
  colors: {
    primaryColor: '#0D47A1',
    secondaryColor: '#f016e1'
  },

  /* Custom fonts for website */
  /*
  fonts: {
    myFont: [
      "Times New Roman",
      "Serif"
    ],
    myOtherFont: [
      "-apple-system",
      "system-ui"
    ]
  },
  */

  // This copyright info is used in /core/Footer.js and blog RSS/Atom feeds.
  copyright: `Copyright © ${new Date().getFullYear()} a-mma`,

  highlight: {
    // The name of the theme used by Highlight.js when highlighting code.
    // You can find the list of supported themes here:
    // https://github.com/isagalaev/highlight.js/tree/master/src/styles
    theme: 'default'
  },

  // Add custom scripts here that would be placed in <script> tags.
  scripts: [
    'https://buttons.github.io/buttons.js',
    {
      src:
        'https://cdnjs.cloudflare.com/ajax/libs/clipboard.js/2.0.0/clipboard.min.js',
      async: true
    }
  ],

  stylesheets: [
    'https://fonts.googleapis.com/css?family=Roboto&display=swap'
  ],

  // On page navigation for the current documentation page.
  onPageNav: 'separate',
  // No .html extensions for paths.
  cleanUrl: true,

  // Open Graph and Twitter card images.
  ogImage: 'img/undraw_online.svg',
  twitterImage: 'img/undraw_tweetstorm.svg',

  // For sites with a sizable amount of content, set collapsible to true.
  // Expand/collapse the links and subcategories under categories.
  // docsSideNavCollapsible: true,

  // Show documentation's last contributor's name.
  // enableUpdateBy: true,

  // Show documentation's last update time.
  // enableUpdateTime: true,

  // You may provide arbitrary config keys to be used as needed by your
  // template. For example, if you need your repo's URL...
  repoUrl: 'https://github.com/a-mma/AquilaDB',
  //editUrl: 'https://github.com/a-mma/AquilaDB/wiki/'
  language:''
};

module.exports = siteConfig;
