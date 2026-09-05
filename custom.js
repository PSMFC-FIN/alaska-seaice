<script>
document.addEventListener("DOMContentLoaded", function() {
  // List of URLs with target=_blank
  const urlsToOpenInNewTab = [
    "https://github.com/polarwatch/alaska-seaice",
    "https://shinyfin.psmfc.org/ak-sst-mhw/",
    "https://polarwatch.noaa.gov"
  ];

  urlsToOpenInNewTab.forEach(url => {
    const link = document.querySelector(`a[href="${url}"]`);
    if (link) {
      link.setAttribute('target', '_blank');
    }
  });

  // Add dismiss button to all tip callouts
  document.querySelectorAll('.callout.callout-tip').forEach(function(callout) {
    if (callout.querySelector('.callout-close')) return;

    const header = callout.querySelector('.callout-header');
    if (!header) return;

    const closeButton = document.createElement('button');
    closeButton.type = 'button';
    closeButton.className = 'callout-close';
    closeButton.setAttribute('aria-label', 'Close tip');
    closeButton.innerHTML = '&times;';
    closeButton.addEventListener('click', function() {
      callout.style.opacity = '0';
      callout.style.transition = 'opacity 0.25s ease';
      setTimeout(function() {
        callout.style.display = 'none';
      }, 250);
    });

    header.appendChild(closeButton);
  });
});
</script>

