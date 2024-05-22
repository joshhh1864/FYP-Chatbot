document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');
  
    form.addEventListener('submit', function (event) {
      event.preventDefault();
  
      const email = form.querySelector('input[placeholder="Your Email"]').value;
      const password = form.querySelector('input[placeholder="Password"]').value;

      const userData = {
        email: email,
        password: password
      };
  
      fetch('/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(userData)
      })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok ' + response.statusText);
        }
        return response.json();
      })
      .then(data => {
        console.log('Success:', data);
        window.location.href = '/chatbot';
      })
      .catch((error) => {
        console.error('Error:', error);
      });
    });
  });
  