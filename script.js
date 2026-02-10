const cards = document.querySelectorAll('.project-card');

cards.forEach((card, index) => {
  card.style.animation = `fadeUp 0.5s ease ${index * 0.08}s both`;
});

const style = document.createElement('style');
style.textContent = `
  @keyframes fadeUp {
    from {
      opacity: 0;
      transform: translateY(8px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
`;
document.head.appendChild(style);
