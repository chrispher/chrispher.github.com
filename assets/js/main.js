$(document).ready(function() {
	
	JekyllSearch.init({
		searchInput: document.getElementById("search-field"),
		jsonFile: "/search.json",
		template: "<p> <a href='{url}' title='{desc}'>{title}</a></p>",
		fuzzy: true		
	});

	//animate header in
	//$('.header-inner-wrapper').delay(700).css({display:'block'}).animate({marginTop:'60px', opacity:'1'}, 600);
	$('header').addClass('animated');
	$('header').addClass('bounceInDown')

	// Animate search in
	$('.fa-search').on('click', function() {
		//$('.search__entry').addClass('search-animate').fadeIn(1000);
		$('.search__entry').css({display:'block'}).animate({marginTop:'0px', opacity:'1'}, 400);
		$('#search-field').focus();
	});


});