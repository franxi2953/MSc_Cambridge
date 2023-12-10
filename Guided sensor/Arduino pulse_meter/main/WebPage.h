const char webpage[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <title>Arduino Data Chart</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/uikit/3.9.4/css/uikit.min.css" />
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="uk-flex uk-flex-center uk-flex-middle" style="height: 100vh;">
    <div class="uk-card uk-card-default uk-card-body" style="width: 50%;">
        <canvas id="myChart"></canvas>
    </div>

    <script>
        var ctx = document.getElementById('myChart').getContext('2d');
        var chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({length: 1000}, (_, i) => i + 1),
                datasets: [{
                    label: 'Sensor Data',
                    backgroundColor: 'rgba(0, 123, 255, 0.5)',
                    borderColor: 'rgba(0, 123, 255, 1)',
                    data: []
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });

        function fetchData() {
            fetch('http://%%IP_ADDRESS%%/retrieve')
                .then(response => response.text())
                .then(data => {
                    var dataArray = data.trim().split('\n').map(Number);
                    chart.data.datasets[0].data = dataArray;
                    chart.update();
                })
                .catch(error => console.error('Error fetching data:', error));
        }

        setInterval(fetchData, 1000); // Fetch data every 1000 milliseconds
    </script>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/uikit/3.9.4/js/uikit.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/uikit/3.9.4/js/uikit-icons.min.js"></script>
</body>
</html>
)rawliteral";
