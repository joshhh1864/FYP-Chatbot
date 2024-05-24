document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');
  
    form.addEventListener('submit', function (event) {
      event.preventDefault();
  
      const name = form.querySelector('input[placeholder="Your Name"]').value;
      const email = form.querySelector('input[placeholder="Your Email"]').value;
      const password = form.querySelector('input[placeholder="Password"]').value;
      const repeatPassword = form.querySelector('input[placeholder="Repeat your password"]').value;
  
      if (password !== repeatPassword) {
        alert('Passwords do not match!');
        return;
      }
  
      const userData = {
        name: name,
        email: email,
        password: password
      };
  
      fetch('/register/new_user', {
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
        window.location.href = '/';
      })
      .catch((error) => {
        console.error('Error:', error);
      });
    });
  });
  