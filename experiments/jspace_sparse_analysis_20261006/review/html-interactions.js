
(() => {
  'use strict';
  document.documentElement.classList.add('js-ready');
  const panels = Array.from(document.querySelectorAll('[data-recall-panel]'));
  const controls = Array.from(document.querySelectorAll('[data-k]'));
  function selectCap(cap) {
    panels.forEach(panel => { panel.hidden = panel.dataset.recallPanel !== cap; });
    controls.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.k === cap)));
  }
  controls.forEach(button => button.addEventListener('click', () => selectCap(button.dataset.k)));
  selectCap('25');
  const dialog = document.getElementById('figure-dialog');
  const zoomImage = document.getElementById('zoom-image');
  const heading = document.getElementById('zoom-heading');
  const closeButton = document.getElementById('close-dialog');
  let previousFocus = null;
  document.querySelectorAll('.figure-zoom').forEach(button => {
    button.addEventListener('click', () => {
      if (typeof dialog.showModal !== 'function') return;
      const source = button.querySelector('img');
      previousFocus = button;
      zoomImage.src = source.src;
      zoomImage.alt = source.alt;
      heading.textContent = button.dataset.title;
      dialog.showModal();
      closeButton.focus();
    });
  });
  closeButton.addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  });
  dialog.addEventListener('close', () => {
    zoomImage.removeAttribute('src');
    if (previousFocus) previousFocus.focus();
  });
  document.getElementById('print-report').addEventListener('click', () => window.print());
  let printDetails = [];
  window.addEventListener('beforeprint', () => {
    printDetails = Array.from(document.querySelectorAll('details')).map(detail => [detail, detail.open]);
    printDetails.forEach(([detail]) => { detail.open = true; });
    document.querySelectorAll('figure img').forEach(img => { img.loading = 'eager'; });
  });
  window.addEventListener('afterprint', () => {
    printDetails.forEach(([detail, wasOpen]) => { detail.open = wasOpen; });
  });
  document.getElementById('download-data').addEventListener('click', () => {
    const blob = new Blob([document.getElementById('reviewed-data').textContent], {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url; link.download = 'reviewed-task-summary.json';
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  if ('IntersectionObserver' in window) {
    const navigation = Array.from(document.querySelectorAll('nav a'));
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        navigation.forEach(link => {
          if (link.hash === '#' + entry.target.id) link.setAttribute('aria-current','location');
          else link.removeAttribute('aria-current');
        });
      }
    }, {rootMargin:'0px 0px -65% 0px'});
    document.querySelectorAll('header#overview, main > section').forEach(section => observer.observe(section));
  }
})();
