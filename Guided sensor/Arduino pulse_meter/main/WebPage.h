const char webpage[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <title>Arduino Data Chart</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/uikit/3.9.4/css/uikit.min.css" />
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/dspjs@1.0.0/dsp.min.js"></script>

</head>
<body class="uk-container">
    <div class="uk-grid">
        <div class="uk-width-1-2 uk-margin-medium-bottom uk-margin-medium-top">
            <div class="uk-card uk-card-default uk-card-body">
                <canvas id="myChart"></canvas>
            </div>
        </div>
        <div class="uk-width-1-2 uk-margin-medium-bottom uk-margin-medium-top">
            <div class="uk-card uk-card-default uk-card-body">
                <canvas id="fftChart"></canvas>
            </div>
        </div>
        <div class="uk-width-1-1">
            <button id="toggleButton" class="uk-button uk-button-default" onclick="toggleDataFetching()">Run</button>
            <button id="downloadButton" class="uk-button uk-button-default" onclick="downloadData()">Download data</button>
        </div>
    </div>

    <script>
      // Define your plugin
      const peakCountPlugin = {
        id: 'peakCountPlugin',
        beforeDraw: function(chart, args, options) {
            const ctx = chart.ctx;
            const width = chart.width;
            const height = chart.height;
            const peakCount = options.peakCount;

            // Calculate duration and BPM
            const timeValues = chart.data.datasets[0].data.map(point => point.x);
            const durationInSeconds = Math.max(...timeValues) - Math.min(...timeValues);
            const beatsPerMinute = (peakCount / durationInSeconds) * 60;

            ctx.restore();
            ctx.font = "20px Arial";
            ctx.fillStyle = "blue";
            ctx.textBaseline = "top";
            ctx.textAlign = "right";

            const text = "BPM: " + beatsPerMinute.toFixed(2); // Display BPM with two decimal places
            const textX = width - 10;
            const textY = 10;

            ctx.fillText(text, textX, textY);
            ctx.save();
        }
    };
      // Register the plugin
      Chart.register(peakCountPlugin);
      
      var chartData = [];
      var updateChart = false; // Initially set to false, will be true when 'Run' is clicked
      var peakCount = 0; // Count of detected peaks
      var ctx = document.getElementById('myChart').getContext('2d');
      var chart = new Chart(ctx, {
          type: 'line',
          data: {
              datasets: [{
                  label: 'Sensor Data',
                  backgroundColor: 'white',
                  borderColor: 'rgba(0, 123, 255, 1)',
                  data: chartData, // Ensure this array is properly populated
                  showLine: true,
                  pointStyle: 'circle',
                  pointRadius: function(context) {
                      var index = context.dataIndex;
                      var value = context.dataset.data[index];
                      // Check if value is defined and has pointRadius property
                      return value && value.pointRadius !== undefined ? value.pointRadius : 0;
                  },
                  pointBackgroundColor: function(context) {
                      var index = context.dataIndex;
                      var value = context.dataset.data[index];
                      // Check if value is defined and has pointBackgroundColor property
                      return value && value.pointBackgroundColor !== undefined ? value.pointBackgroundColor : 'rgba(0, 123, 255, 1)';
                  }
              }]
          },
          options: {
              animation: false,
              legend: {
                  display: false // Hide the legend
              },
              scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Amplitude'
                    }
                },
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: 'Time (s)'
                    }
                }
            },
              plugins: {
                peakCountPlugin: {
                  peakCount: 0 // Initialize peak count
                },
                legend: {
                    display: false // This will hide the legend
                },
              }
            }
        });

        // Setup for the FFT chart
        var fftCtx = document.getElementById('fftChart').getContext('2d');
        var fftChart = new Chart(fftCtx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'FFT Analysis',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    data: [], // FFT data goes here
                    showLine: true,
                }]
            },
            options: {
              animation: false,
              legend: {
                  display: false // Hide the legend
              },
              scales: {
                  y: {
                      beginAtZero: true,
                      title: {
                          display: true,
                          text: 'Ampluitude'
                      }
                  },
                   x: {
                      type: 'linear',
                      position: 'bottom',
                      title: {
                          display: true,
                          text: 'Freq (Hz)'
                      },
                      min: 0.5 // Set minimum frequency to 0.5 Hz
                  }
              },
              plugins: {
                legend: {
                    display: false // This will hide the legend
                },
              }
            }
        });


        function padArrayToNextPowerOfTwo(array) {
            var length = array.length;
            var nextPowerOfTwo = Math.pow(2, Math.ceil(Math.log2(length)));
            return array.concat(new Array(nextPowerOfTwo - length).fill(0));
        }


    function performFFT(dataArray) {
        // Ensure dataArray is not empty
        if (dataArray.length === 0) {
            console.error('Data array is empty');
            return [];
        }

        var signal = dataArray.map(point => point.y);

        // Calculate and subtract the average (DC offset) from the signal
        var average = signal.reduce((acc, val) => acc + val, 0) / signal.length;
        signal = signal.map(val => val - average);

        var timeValues = dataArray.map(point => point.x);
        var totalTimeSpan = Math.max(...timeValues) - Math.min(...timeValues);

        if (totalTimeSpan <= 0) {
            console.error('Invalid time span');
            return [];
        }

        var samplingRate = signal.length / totalTimeSpan;
        var fftSize = Math.pow(2, Math.ceil(Math.log2(signal.length)));
        
        // Ensure the signal length matches the fftSize
        if (signal.length < fftSize) {
            console.error('Signal length is smaller than fftSize. Padding with zeros.');
            while (signal.length < fftSize) {
                signal.push(0);
            }
        } else if (signal.length > fftSize) {
            console.error('Signal length is larger than fftSize. Truncating.');
            signal = signal.slice(0, fftSize);
        }

        var fft = new FFT(fftSize, samplingRate);
        fft.forward(signal);

        var fftData = [];
        for (let index = 0; index < fft.spectrum.length; index++) {
            var frequency = index * (samplingRate / fftSize);
            var value = fft.spectrum[index];

            // Filter for pulse rate frequencies and convert frequency to BPM
            if (frequency >= 0.67 && frequency <= 10) {
                fftData.push({ x: frequency, y: value }); 
            }
        }
        return fftData;
    }



  function updateFFTData() {
      var fftData = performFFT(chartData);

      // Find the highest peak within the pulse rate frequency range
      var highestPeak = { frequency: 0, value: 0 };
      fftData.forEach(point => {
          // Update if the current point has a higher amplitude
          // and is within the acceptable frequency range for pulse rate
          if (point.y > highestPeak.value && point.x >= 0.5 && point.x <= 3.5) {
              highestPeak.frequency = point.x;
              highestPeak.value = point.y;
          }
      });

      // Calculate BPM from the highest peak frequency
      var beatsPerMinute = highestPeak.frequency * 60; // Convert Hz to BPM

      // Update the FFT chart with the new data and display the calculated BPM
      fftChart.options.plugins.peakCountPlugin = {
          peakCount: beatsPerMinute.toFixed(2) // Display BPM with two decimal places
      };
      fftChart.data.datasets[0].data = fftData;
      fftChart.update();
  }



        var fetchDataInterval;
          function toggleDataFetching() {
            var button = document.getElementById('toggleButton');
            if (button.textContent === 'Run') {
                updateChart = true;
                fetchData(); // Start fetching data
                button.textContent = 'Stop';
            } else {
                updateChart = false;
                button.textContent = 'Run';
            }
          }

        function fetchData() {
            if (!updateChart) return; // Stop fetching if updateChart is false

            fetch('http://%%IP_ADDRESS%%/retrieve')
                .then(response => response.text())
                .then(data => {
                    var lines = data.trim().split('\n');
                    chartData = lines.map(line => {
                        var parts = line.split(',');
                        return {
                            x: parseFloat(parts[0]),
                            y: parseFloat(parts[1]),
                            pointBackgroundColor: 'rgba(0, 123, 255, 1)', // Default point color
                            pointRadius: 0, // Default point radius
                            pointStyle: 'circle' // Default point style
                        };
                    });

                    chartData = detectPeaks(chartData); // Call the peak detection function

                    // Update the peakCount in the chart's options before updating the chart
                    chart.options.plugins.peakCountPlugin.peakCount = peakCount;

                    chart.data.datasets[0].data = chartData;
                    if (updateChart) chart.update();
                    fetchData(); // Fetch the next set of data
                })
                .catch(error => {
                    console.error('Error fetching data:', error);
                    fetchData(); // Attempt to fetch data again
                });

          updateFFTData(); // Update FFT chart after updating the main chart
        }

        function detectPeaks(data) {
          peakCount = 0; // Reset peak count
          const len = 50; // Length for running average
          const beatsLen = 10; // Length for interval average
          const tolerance = 0.5;
          let stored = new Array(len).fill(0); // Initialize with zeros
          let beats = new Array(beatsLen).fill(0); // Initialize with zeros
          let beatSum = 0;
          let skip = false;

          for (let i = 0; i < data.length; i++) {
              let input = data[i].y;
              stored.shift(); // Remove the oldest element
              stored.push(input); // Push the new input

              // Calculate running average
              let sum = stored.reduce((acc, val) => acc + val, 0) / len;

              beatSum = beats.reduce((acc, val) => acc + val, 0) / beatsLen;
              if ((input > (beatSum * (tolerance + 1)) || input < (beatSum * (1 - tolerance))) && !skip) {
                  skip = true;
                  data[i].pointBackgroundColor = 'red'; // Mark as peak
                  data[i].pointRadius = 5;
                  peakCount++;
              } else {
                  beats.shift(); // Remove the oldest interval
                  beats.push(input); // Push the new interval
                  skip = false;
                  data[i].pointBackgroundColor = 'rgba(0, 123, 255, 1)'; // Default style
                  data[i].pointRadius = 0;
              }
          }
          return data;
      }


        function downloadData() {
            var csvContent = "data:text/csv;charset=utf-8,";
            csvContent += "Time,Value\n";
            chartData.forEach(function(rowArray) {
                var row = rowArray.x + "," + rowArray.y;
                csvContent += row + "\r\n";
            });

            var encodedUri = encodeURI(csvContent);
            var link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", "chart_data.csv");
            document.body.appendChild(link); // Required for FF

            link.click(); // This will download the data file named "chart_data.csv".
        }
    </script>


    <script src="https://cdnjs.cloudflare.com/ajax/libs/uikit/3.9.4/js/uikit.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/uikit/3.9.4/js/uikit-icons.min.js"></script>
</body>
</html>
)rawliteral";
