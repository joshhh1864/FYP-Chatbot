function showNotification(message, type = 'success') {
  const container = document.getElementById('notification-container');

  // Create notification element
  const notification = document.createElement('div');
  notification.classList.add('notification', type);
  notification.innerHTML = `
    <span>${message}</span>
    <span class="close-btn">&times;</span>
  `;

  // Append notification to container
  container.appendChild(notification);

  // Show notification
  setTimeout(() => {
    notification.classList.add('show');
  }, 10);

  // Remove notification after 3 seconds
  setTimeout(() => {
    notification.classList.remove('show');
    setTimeout(() => {
      container.removeChild(notification);
    }, 500);
  }, 3000);

  // Add close button event
  notification.querySelector('.close-btn').addEventListener('click', () => {
    notification.classList.remove('show');
    setTimeout(() => {
      container.removeChild(notification);
    }, 500);
  });
}


document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');
  
    form.addEventListener('submit', function (event) {
      event.preventDefault();
  
      const name = form.querySelector('input[placeholder="Your Name"]').value;
      const email = form.querySelector('input[placeholder="Your Email"]').value;
      const password = form.querySelector('input[placeholder="Password"]').value;
      const repeatPassword = form.querySelector('input[placeholder="Repeat your password"]').value;
  
      if (password !== repeatPassword) {
        showNotification('Passwords do not match!', 'error');
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
          showNotification('Error in Registration!', 'error');
          throw new Error('Network response was not ok ' + response.statusText);
        }
        return response.json();
      })
      .then(data => {
        showNotification('Registration successful!', 'success');
        console.log('Success:', data);
        window.location.href = '/';
      })
      .catch((error) => {
        console.error('Error:', error);
      });
    });
  });


  