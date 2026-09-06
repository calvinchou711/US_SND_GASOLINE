<!doctype html>
<html <?php language_attributes(); ?>>
<head>
	<meta charset="<?php bloginfo( 'charset' ); ?>">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<?php wp_head(); ?>
</head>
<body <?php body_class( 'portfolio-index-page' ); ?>>
<?php wp_body_open(); ?>
<a class="skip-link landing-skip-link" href="#main">Skip to content</a>

<div class="portfolio-index">
	<nav class="landing-nav" aria-label="Primary navigation">
		<a href="<?php echo esc_url( home_url( '/portfolio/' ) ); ?>" aria-current="page">Portfolio</a>
		<a href="https://www.linkedin.com/in/calvin-chou/" target="_blank" rel="noopener noreferrer">LinkedIn</a>
		<a href="https://github.com/calvinchou711" target="_blank" rel="noopener noreferrer">GitHub</a>
	</nav>

	<main class="portfolio-main" id="main">
		<h1 class="portfolio-heading">Portfolio</h1>
		<div class="portfolio-projects">
			<a href="<?php echo esc_url( home_url( '/oil-market-monitor/' ) ); ?>">US Oil Dashboard</a>
			<a href="<?php echo esc_url( home_url( '/portfolio/us-supply-demand-model-crude/' ) ); ?>">US Supply &amp; Demand Model for Crude</a>
			<a href="<?php echo esc_url( home_url( '/portfolio/us-supply-demand-model-gasoline/' ) ); ?>">US Supply &amp; Demand Model for Total Gasoline</a>
		</div>
	</main>
</div>

<?php wp_footer(); ?>
</body>
</html>
