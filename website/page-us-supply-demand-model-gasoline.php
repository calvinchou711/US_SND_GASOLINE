<!doctype html>
<html <?php language_attributes(); ?>>
<head>
	<meta charset="<?php bloginfo( 'charset' ); ?>">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<?php wp_head(); ?>
</head>
<body <?php body_class( 'project-detail-page' ); ?>>
<?php wp_body_open(); ?>
<a class="skip-link" href="#main">Skip to content</a>

<header class="project-detail__top">
	<a href="<?php echo esc_url( home_url( '/portfolio/' ) ); ?>">&larr; Portfolio</a>
	<a href="https://github.com/calvinchou711/US_SND_GASOLINE" target="_blank" rel="noopener noreferrer">GitHub &nearr;</a>
</header>

<main class="project-detail__main" id="main">
	<p class="project-detail__label">Project</p>
	<h1>US Supply &amp; Demand Model for Total Gasoline</h1>

	<p>Finished motor gasoline plus motor gasoline blending components, modeled across all five U.S. PADDs.</p>

	<section class="project-detail__notebook" aria-labelledby="notebook-heading">
		<div class="project-detail__notebook-head">
			<h2 id="notebook-heading">Total gasoline results notebook</h2>
			<a href="<?php echo esc_url( home_url( '/project-notebooks/gasoline/us_snd_model_results.ipynb' ) ); ?>" download>Download .ipynb &darr;</a>
		</div>
		<iframe class="project-detail__frame" src="<?php echo esc_url( home_url( '/project-notebooks/gasoline/us_snd_model_results.html?v=total-gasoline-20260905' ) ); ?>" title="US total gasoline supply and demand model results notebook"></iframe>
	</section>
</main>

<?php wp_footer(); ?>
</body>
</html>
