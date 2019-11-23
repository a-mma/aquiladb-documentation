/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const React = require('react');
const CompLibrary = require('../../core/CompLibrary.js');
const MarkdownBlock = CompLibrary.MarkdownBlock; /* Used to read markdown */
const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

class HomeSplash extends React.Component {
  render() {
    const { siteConfig, language = '' } = this.props;
    const { baseUrl, docsUrl } = siteConfig;
    const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
    const langPart = `${language ? `${language}/` : ''}`;
    const docUrl = doc => `${baseUrl}${docsPart}${langPart}${doc}`;

    const SplashContainer = props => (
      <div className="homeContainer">
        <div className="homeSplashFade">
          <div className="wrapper homeWrapper">{props.children}</div>
        </div>
      </div>
    );

    const Logo = props => (
      <div className="projectLogo">
        <img src={props.img_src} alt="Project Logo" />
      </div>
    );

    const ProjectTitle = () => (
      <h2 className="projectTitle">
        {siteConfig.title}
        <small>{siteConfig.tagline}</small>
      </h2>
    );

    const PromoSection = props => (
      <div className="section promoSection">
        <div className="promoRow">
          <div className="pluginRowBlock">{props.children}</div>
        </div>
      </div>
    );

    const Button = props => (
      <div className="pluginWrapper buttonWrapper">
        <a className="button" href={props.href} target={props.target}>
          {props.children}
        </a>
      </div>
    );

    return (
      <SplashContainer>
        {/* <Logo img_src={`${baseUrl}img/undraw_monitor.svg`} /> */}
        <div className="inner">
          <ProjectTitle siteConfig={siteConfig} />
          <PromoSection>
            <Button href="#try">Try It Out</Button>
            <Button href="http://www.youtube.com/watch?v=-VYpjpLXU5Q">
              Media
            </Button>
            <Button href="http://bit.ly/faya-slideshare">Slides</Button>
          </PromoSection>
        </div>
      </SplashContainer>
    );
  }
}

class Index extends React.Component {
  render() {
    const { config: siteConfig, language = '' } = this.props;
    const { baseUrl } = siteConfig;
    const Block = props => (
      <Container
        padding={['bottom', 'top']}
        id={props.id}
        background={props.background}
      >
        <GridBlock
          align="center"
          contents={props.children}
          layout={props.layout}
        />
      </Container>
    );

    const FeatureCallout = () => (
      <div
        className="productShowcaseSection paddingBottom"
        style={{ textAlign: 'center' }}
      >
        <h2>Feature Callout</h2>
      </div>
    );

    const TryOut = props => (
      <div>
        <Block id="try">
          {[
            {
              content:
                'To make your landing page more attractive, use illustrations! Check out ' +
                '[**unDraw**](https://undraw.co/) which provides you with customizable illustrations which are free to use. ' +
                'The illustrations you see on this page are from unDraw.',
              image: `${baseUrl}img/undraw_code_review.svg`,
              imageAlign: 'left',
              title: 'Get Started'
            }
          ]}
        </Block>
      </div>
    );

    const Description = () => (
      <Block background="dark">
        {[
          {
            content:
              'AquilaDB is a latent vector and document database built for Data Scientists and Machine Learning engineers. Prototype an idea in minutes then scale at your pace. AquilaDB is the Redis for Machine Learning',
            image: `${baseUrl}img/machine-learning.svg`,
            imageAlign: 'right',
            title: "A bird's eye-view"
          }
        ]}
      </Block>
    );

    const LearnHow = () => (
      <Block background="light">
        {[
          {
            content:
              'Each new Docusaurus project has **randomly-generated** theme colors.',
            image: `${baseUrl}img/undraw_youtube_tutorial.svg`,
            imageAlign: 'right',
            title: 'Randomly Generated Theme Colors'
          }
        ]}
      </Block>
    );

    const Features = () => (
      <Block layout="fourColumn">
        {[
          {
            content: `Efficient similarity search on dense vectors of any size that do not fit in RAM. 
            Native to your ML model's internal representation`,
            image: `${baseUrl}img/neural.svg`,
            imageAlign: 'top',
            title: 'Redis of Machine Learning'
          },
          {
            content: `Setup and start prototyping your idea in minutes and scale at you pace.
               In many cases, all you need is AquilaDB and pre-trained ML model`,
            image: `${baseUrl}img/socket.svg`,
            imageAlign: 'top',
            title: 'Easy to start'
          },
          {
            content:
              'Be in your existing environment and continue working with you environment. Be in your existing environment and continue working with you environment',
            image: `${baseUrl}img/browser.svg`,
            imageAlign: 'top',
            title: 'Language'
          },
          {
            content: `Take advantage of everything that <a href="https://docs.couchdb.org/en/master/replication/protocol.html" target="_blank">Couch protocol</a> can offer. Native connectivity to anything that understands couch protocol`,
            image: `${baseUrl}img/box.svg`,
            imageAlign: 'top',
            title: 'Replication and Scaling'
          }
        ]}
      </Block>
    );

    const Showcase = () => {
      if ((siteConfig.users || []).length === 0) {
        return null;
      }

      const showcase = siteConfig.users
        .filter(user => user.pinned)
        .map(user => (
          <a href={user.infoLink} key={user.infoLink}>
            <img src={user.image} alt={user.caption} title={user.caption} />
          </a>
        ));

      const pageUrl = page => baseUrl + (language ? `${language}/` : '') + page;

      return (
        <div className="productShowcaseSection paddingBottom">
          <h2>Who is Using This?</h2>
          <p>This project is used by all these people</p>
          <div className="logos">{showcase}</div>
          <div className="more-users">
            <a className="button" href={pageUrl('users.html')}>
              More {siteConfig.title} Users
            </a>
          </div>
        </div>
      );
    };

    return (
      <div>
        <HomeSplash siteConfig={siteConfig} language={language} />
        <div className="mainContainer">
          <Features />
          <Description />
          <TryOut/>
        </div>
      </div>
    );
  }
}

module.exports = Index;
