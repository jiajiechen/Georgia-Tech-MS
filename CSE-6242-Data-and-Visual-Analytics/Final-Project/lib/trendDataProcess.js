function userExposureMonitor(phaseTime, phaseNum, campInterval, campNum, nodes, memoryLength, simLength, userNum){
	var data = new Array(userNum);
	var startTime;
	var timer, user;
	for (var i = 0; i<userNum; i++){
		data[i] = new Array(simLength).fill(0);
	}
	for (var camp = 0; camp<campNum; camp++){
		for (user = 0; user<userNum; user++){
			if (nodes[user].phase1==1){
				phase = 0;
				startTime = campInterval*camp+phaseTime*phase;
				for (timer = startTime; timer<startTime+memoryLength; timer++ ){
					data[user][timer] = 1;
				}
			}else if (nodes[user].phase2==1){
				phase = 1;
				startTime = campInterval*camp+phaseTime*phase;
				for (timer = startTime; timer<startTime+memoryLength; timer++ ){
					data[user][timer] = 1;
				}
			}else if (nodes[user].phase3==1){
				phase = 2;
				startTime = campInterval*camp+phaseTime*phase;
				for (timer = startTime; timer<startTime+memoryLength; timer++ ){
					data[user][timer] = 1;
				}
			}								
		}
	}
	var userNumLog = new Array(simLength).fill(0);
	for (var time = 0; time<simLength; time++){
		for (user = 0; user<userNum; user++){
			userNumLog[time] = userNumLog[time]+data[user][time];
		}		
	}

	return userNumLog;
}



function createUserExposureJSON(userNumLog,simLength){
	var dataJSON = [];
	var time, day, hour, minute, dayString, hourString, minuteString;
	for (var i = 0; i<simLength; i++){
		day = Math.floor(i/(60*24))+1;
		hour = Math.floor(i%(60*24)/60);
		minute = Math.floor(i%60);
		if (day<10){dayString = '0'+day.toString();} else {dayString = day.toString();}
		if (hour<10){hourString = '0'+hour.toString();} else {hourString = hour.toString();}
		if (minute<10){minuteString = '0'+minute.toString();} else {minuteString = minute.toString();}
		//timeFormat.parse('2014-03-08T12:00:00')
		dataJSON.push({
			date : '2016-01-'+dayString+'T'+hourString+':'+ hourString +':00',
			exposedUserNum : userNumLog[i]
		});
	}
	return dataJSON;
}

function createMarkersJSON(campInterval,campNum){
	var dataJSON = [];
	var time, day, hour, minute, dayString, hourString, minuteString;
	for (var i = 0; i<campNum; i++){
		time = i*campInterval;
		day = Math.floor(time/(60*24))+1;
		hour = Math.floor(time%(60*24)/60);
		minute = Math.floor(time%60);
		if (day<10){dayString = '0'+day.toString();} else {dayString = day.toString();}
		if (hour<10){hourString = '0'+hour.toString();} else {hourString = hour.toString();}
		if (minute<10){minuteString = '0'+minute.toString();} else {minuteString = minute.toString();}
		dataJSON.push({
			date : '2016-01-'+dayString+'T'+hourString+':'+ hourString +':00',
			campID : 'Camp. ' + (i+1).toString()
		});
	}
	return dataJSON;
}




// start drawing functions


function addAxesAndLegend (svg, xAxis, yAxis, margin, chartWidth, chartHeight) {
  var legendWidth  = 300,
      legendHeight = 40;
  var paddle = {left: 50};
  // clipping to make sure nothing appears behind legend
  svg.append('clipPath')
    .attr('id', 'axes-clip')
    .append('polygon')
    .attr('points', (-margin.left)                 + ',' + (-margin.top)                 + ' ' +
                      (chartWidth - legendWidth - 1 ) + ',' + (-margin.top)                 + ' ' +
                      (chartWidth - legendWidth - 1 ) + ',' + legendHeight                  + ' ' +
                      (chartWidth + margin.right)    + ',' + legendHeight                  + ' ' +
                      (chartWidth + margin.right)    + ',' + (chartHeight + margin.bottom) + ' ' +
                      (-margin.left)                 + ',' + (chartHeight + margin.bottom));

  var axes = svg.append('g')
    .attr('clip-path', 'url(#axes-clip)');

  axes.append('g')
    .attr('class', 'x axis')
    .attr('transform', 'translate(0,' + chartHeight + ')')
    .call(xAxis);

  axes.append('g')
    .attr('class', 'y axis')
    .attr('transform', 'translate('+paddle.left+',0)')
    .call(yAxis)
    .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 6)
      .attr('dy', '.71em')
      .style('text-anchor', 'end')
      .text('User Number');

  var legend = svg.append('g')
    .attr('class', 'legend')
    .attr('transform', 'translate(' + (chartWidth - legendWidth) + ', 0)');

  legend.append('rect')
    .attr('class', 'legend-bg')
    .attr('width',  legendWidth)
    .attr('height', legendHeight);

  legend.append('path')
    .attr('class', 'median-line')
    .attr('d', 'M10,20L85,20');

  legend.append('text')
    .attr('x', 150)
    .attr('y', 25)
    .text('Exposed User Number');
}

function drawPaths (svg, data, x, y) {
  var oneLine = d3.svg.line()
    .interpolate('basis-open')
    .x(function (d) { return x(d.date); })
    .y(function (d) { return y(d.exposedUserNum); });

  var lowerArea = d3.svg.area()
    .interpolate('basis-open')
    .x (function (d) { return x(d.date) || 1; })
    .y0(function (d) { return y(d.exposedUserNum); })
    .y1(function (d) { return y(d.base); });


  svg.datum(data);

  svg.append('path')
    .attr('class', 'area inner')
    .attr('d', lowerArea)
    .attr('clip-path', 'url(#rect-clip)');

  svg.append('path')
    .attr('class', 'median-line')
    .attr('d', oneLine)
    .attr('clip-path', 'url(#rect-clip)');
}

function addMarker (marker, svg, chartHeight, x) {
  var radius = 32,
      xPos = x(marker.date) - radius ,
      yPosStart = chartHeight - radius - 3,
      yPosEnd = 50 + radius - 3;

  var markerG = svg.append('g')
    .attr('class', 'marker '+marker.campID.toLowerCase())
    .attr('transform', 'translate(' + xPos + ', ' + yPosStart + ')')
    .attr('opacity', 0);

  markerG.transition()
    .duration(1000)
    .attr('transform', 'translate(' + xPos + ', ' + yPosEnd + ')')
    .attr('opacity', 1);

  markerG.append('path')
    .attr('d', 'M' + radius + ',' + (chartHeight-yPosStart) + 'L' + radius + ',' + (chartHeight-yPosStart))
    .transition()
      .duration(1000)
      .attr('d', 'M' + radius + ',' + (chartHeight-yPosEnd) + 'L' + radius + ',' + (radius*2));

  markerG.append('circle')
    .attr('class', 'marker-bg')
    .attr('cx', radius)
    .attr('cy', radius)
    .attr('r', radius);

  markerG.append('text')
    .attr('x', radius)
    .attr('y', radius*1.1)
    .text(marker.campID);
}





function makeChart (svg3, data, markers, svgWidth, svgHeight, campTime,simLength) {
  var svg = svg3;
  var margin = { top: 20, right: 20, bottom: 40, left: 40 },
      chartWidth  = svgWidth  - margin.left - margin.right,
      chartHeight = svgHeight - margin.top  - margin.bottom;
  var paddle = {left: 50};

  var x = d3.time.scale().range([paddle.left, chartWidth])
            .domain(d3.extent(data, function (d) { return d.date; })),
      y = d3.scale.linear().range([chartHeight, 0])
            .domain([0, 150]);
//d3.max(data, function (d) { return d.exposedUserNum; })*1.2
  var xAxis = d3.svg.axis().scale(x).orient('bottom')
                .innerTickSize(-chartHeight).outerTickSize(0).tickPadding(10),
      yAxis = d3.svg.axis().scale(y).orient('left')
                .innerTickSize(-chartWidth).outerTickSize(0).tickPadding(10);

  svg.append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');


  // clipping to start chart hidden and slide it in later
  var rectClip = svg.append('clipPath')
    .attr('id', 'rect-clip')
    .append('rect')
      .attr('width', 0)
      .attr('height', chartHeight);

  addAxesAndLegend(svg, xAxis, yAxis, margin, chartWidth, chartHeight);
  drawPaths(svg, data, x, y);
  startTransitions(svg, chartWidth, chartHeight, rectClip, markers, x, campTime,simLength);
}


function startTransitions (svg, chartWidth, chartHeight, rectClip, markers, x, campTime,simLength) {

  rectClip.transition()
    .ease("linear")
    //.delay(-12000)
    .delay(-2500)
    .duration(simLength/60*2000*1.05)
    .attr('width', chartWidth);

  markers.forEach(function (marker, i) {
    setTimeout(function () {
      addMarker(marker, svg, chartHeight, x);
    }, 1000 + 500*i);
  });
}





