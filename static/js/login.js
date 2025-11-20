// login.js - simple form switcher for overlay form
(function(){
  function byId(id){return document.getElementById(id)}
  var container = document.getElementById('container');
  // buttons
  var signUpBtn = byId('signUp');
  var signInBtn = byId('signIn');
  var mobileToSignUp = byId('mobileSwitchToSignUp');
  var mobileToSignIn = byId('mobileSwitchToSignIn');

  function addListeners(){
    if(signUpBtn) signUpBtn.addEventListener('click', function(){ container.classList.add('right-panel-active'); });
    if(signInBtn) signInBtn.addEventListener('click', function(){ container.classList.remove('right-panel-active'); });
    if(mobileToSignUp) mobileToSignUp.addEventListener('click', function(e){ e.preventDefault(); container.classList.add('right-panel-active'); window.scrollTo({top:0,behavior:'smooth'}); });
    if(mobileToSignIn) mobileToSignIn.addEventListener('click', function(e){ e.preventDefault(); container.classList.remove('right-panel-active'); window.scrollTo({top:0,behavior:'smooth'}); });
  }

  if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', addListeners);
  else addListeners();
})();
