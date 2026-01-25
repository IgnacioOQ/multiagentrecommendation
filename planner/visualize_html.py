#!/usr/bin/env python3
import sys
import os
import json
import webbrowser
import base64
import urllib.request

# Setup path to find 'language' module
current_file_path = os.path.abspath(__file__)
planner_dir = os.path.dirname(current_file_path)
manager_dir = os.path.dirname(planner_dir)
root_dir = os.path.dirname(manager_dir)
# Setup path to find 'language' module
current_file_path = os.path.abspath(__file__)
planner_dir = os.path.dirname(current_file_path)
language_dir = os.path.join(planner_dir, 'language')

if language_dir not in sys.path:
    sys.path.append(language_dir)

try:
    from md_parser import MarkdownParser
except ImportError as e:
    print(f"Error: Could not import md_parser from {language_dir}. {e}")
    sys.exit(1)

# Ensure D3 is available locally
D3_URL = "https://cdn.jsdelivr.net/npm/d3@7.8.5/dist/d3.min.js"
D3_PATH = os.path.join(planner_dir, "d3.min.js")

def ensure_d3():
    if not os.path.exists(D3_PATH):
        print(f"Downloading D3.js from {D3_URL}...")
        try:
            urllib.request.urlretrieve(D3_URL, D3_PATH)
            print("D3.js downloaded successfully.")
        except Exception as e:
            print(f"Error downloading D3.js: {e}")
            print("Visualization may fail if offline.")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Master Plan Visualization</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7.8.5/dist/d3.min.js"></script>
  <script>
    if (typeof d3 === 'undefined') {
        document.write('<script src="d3.min.js" onerror="handleScriptError()"><\/script>');
    }
  </script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; overflow: hidden; background: #f8f9fa; }
    #container { width: 100vw; height: 100vh; display: flex; }
    #viz { flex-grow: 1; height: 100%; position: relative; }
    #sidebar { width: 400px; height: 100vh; background: white; border-left: 1px solid #ddd; padding: 20px; box-sizing: border-box; overflow-y: auto; display: none; box-shadow: -2px 0 5px rgba(0,0,0,0.05); z-index: 10; transform: translateX(0); transition: transform 0.3s ease; }
    
    .node circle { fill: #fff; stroke: steelblue; stroke-width: 2px; cursor: pointer; transition: all 0.3s; }
    .node circle:hover { stroke-width: 4px; }
    .node text { font: 12px sans-serif; cursor: pointer; text-shadow: 0 1px 0 #fff, 1px 0 0 #fff, 0 -1px 0 #fff, -1px 0 0 #fff; }
    
    .link { fill: none; stroke: #ccc; stroke-width: 1.5px; transition: all 0.5s; stroke-opacity: 0.6; }

    .dep-link { fill: none; stroke: #e74c3c; stroke-width: 1.5px; stroke-dasharray: 4; marker-end: url(#arrowhead); opacity: 0.6; }
    .dep-link:hover { opacity: 1.0; stroke-width: 2.5px; }

    /* Metadata Colors */
    .status-done { stroke: #2ecc71 !important; fill: #e8f8f5; }
    .status-active, .status-in-progress { stroke: #3498db !important; fill: #ebf5fb; }
    .status-todo { stroke: #bdc3c7 !important; fill: #fbfcfc; }
    .status-blocked { stroke: #e74c3c !important; fill: #fdedec; }
    
    h2 { margin-top: 0; font-size: 1.5em; color: #2c3e50; }
    .meta-tag { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; margin-right: 5px; margin-bottom: 5px; color: white; font-weight: 500;}
    
    .tag-todo { background: #95a5a6; }
    .tag-active, .tag-in-progress { background: #3498db; }
    .tag-done { background: #2ecc71; }
    .tag-blocked { background: #e74c3c; }
    .tag-default { background: #7f8c8d; }

    pre { background: #f4f6f7; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 0.9em; white-space: pre-wrap; }
    .content-block { line-height: 1.6; color: #34495e; font-size: 0.95em; }

    .control-panel { position: absolute; top: 20px; left: 20px; background: rgba(255, 255, 255, 0.9); padding: 10px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    button { background: #3498db; color: white; border: none; padding: 5px 12px; border-radius: 4px; cursor: pointer; font-size: 13px; margin-right: 5px; }
    button:hover { background: #2980b9; }

    #debug-log { position: absolute; bottom: 10px; left: 10px; font-family: monospace; font-size: 10px; color: #aaa; pointer-events: none; z-index: 100; max-height: 200px; overflow: hidden; }
    #loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 1.5em; color: #666; background: rgba(255,255,255,0.8); padding: 20px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
  </style>
</head>
<body>

<div id="container">
  <div id="viz">
    <div id="loading">Initializing...</div>
    <div id="debug-log"></div>
    <svg width="100%" height="100%"></svg>
    <div class="control-panel">
        <button onclick="expandAll()">Expand All</button>
        <button onclick="collapseAll()">Collapse All</button>
        <button onclick="resetZoom()">Reset Zoom</button>
        <button onclick="toggleDeps()">Toggle Dependencies</button>
    </div>
  </div>
  <div id="sidebar">
    <div id="details">
        <h2 style="color: #bbb;">Select a node...</h2>
    </div>
  </div>
</div>

<script>
// Global Variables - Must be declared before use
let root, svg, g, zoom, tree;
let i = 0;
let duration = 500;
let parsedData = null;
let showDependencies = true;

function log(msg) {
    console.log(msg);
    const logDiv = document.getElementById('debug-log');
    if (logDiv) logDiv.innerHTML += msg + "<br>";
}

function handleScriptError() {
    log("ERROR: Failed to load D3.js. Check if d3.min.js exists in the same folder.");
    document.getElementById('loading').innerHTML = "Error: D3.js missing.<br><small>Ensure d3.min.js is in the folder.</small>";
}

// Base64 Decode
function decodeData(enc) {
    try {
        log("Decoding data...");
        const jsonStr = new TextDecoder().decode(Uint8Array.from(atob(enc), c => c.charCodeAt(0)));
        log("Data decoded. Length: " + jsonStr.length);
        return JSON.parse(jsonStr);
    } catch(e) {
        log("Decoding error: " + e.message);
        document.getElementById('loading').innerText = "Error decoding data";
        return null;
    }
}

// Data Injection
try {
    const rawData = "__DATA_PLACEHOLDER__";
    if (rawData.startsWith("__DATA")) {
        log("Error: Placeholder not replaced.");
    } else {
        const data = decodeData(rawData);
        if (data && typeof d3 !== 'undefined') {
            log("D3 loaded. Version: " + d3.version);
            document.getElementById('loading').style.display = 'none';
            parsedData = data.tree ? data : { tree: data, dependencies: [] }; // Handle legacy format if needed
            initViz(parsedData);
        } else {
            if (typeof d3 === 'undefined') {
                handleScriptError();
            } else {
                log("Error: Data is null.");
            }
        }
    }
} catch (globalErr) {
    log("Global Error: " + globalErr.message);
}

function initViz(fullData) {
    log("Initializing Visualization...");
    try {
        const width = window.innerWidth;
        const height = window.innerHeight;

        svg = d3.select("svg");
        
        // Clear previous if any
        svg.selectAll("*").remove();
        
        // Defs for arrows
        const defs = svg.append("defs");
        defs.append("marker")
            .attr("id", "arrowhead")
            .attr("refX", 10) /* adjust based on node radius */
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-2.5 L6,0 L0,2.5")
            .style("fill", "#e74c3c");

        g = svg.append("g");

        zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => g.attr("transform", event.transform));

        svg.call(zoom);

        // Define tree layout params
        // nodeSize depends on orientation. 
        // For standard top-down: [width, height]
        // For left-right (projecting y as x): [height, width]
        tree = d3.tree().nodeSize([40, 300]); 

        root = d3.hierarchy(fullData.tree, d => d.children);
        root.x0 = 0;
        root.y0 = 0;

        // Check if root has children
        if (!root.children) {
            log("Warning: Root has no children.");
        } else {
            log("Root children count: " + root.children.length);
        }

        // Collapse
        if(root.children) {
            root.children.forEach(collapseRecursive);
        }

        update(root);
        
        // Initial center
        const initialTransform = d3.zoomIdentity.translate(100, height / 2).scale(1);
        svg.call(zoom.transform, initialTransform);
        
        log("Viz Initialized.");

    } catch (vizErr) {
        log("Viz Error: " + vizErr.message);
        log(vizErr.stack);
    }
}

function collapseRecursive(d) {
  if(d.children) {
    d._children = d.children;
    d.children.forEach(collapseRecursive);
    d.children = null;
  }
}

function expandAll() {
    function recurse(d) {
        if (d._children) {
            d.children = d._children;
            d._children = null;
        }
        if (d.children) d.children.forEach(recurse);
    }
    recurse(root);
    update(root);
}

function collapseAll() {
    if (root.children) root.children.forEach(collapseRecursive);
    update(root);
}

function resetZoom() {
    const height = window.innerHeight;
    svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity.translate(100, height/2).scale(1));
}

function toggleDeps() {
    showDependencies = !showDependencies;
    update(root);
}

function update(source) {
  const treeData = tree(root);

  // Compute the new tree layout.
  const nodes = treeData.descendants();
  const links = treeData.links();

  // Normalize for left-right tree
  // Swap x and y for horizontal layout
  nodes.forEach(d => { d.y = d.depth * 300; });

  // Node Map for calculating dependencies
  const nodeMap = new Map();
  nodes.forEach(d => {
      // Key can be ID or Node Title (fallback)
      if (d.data.metadata && d.data.metadata.id) {
          nodeMap.set(d.data.metadata.id, d);
      }
      // Also map by title for redundancy if needed, but ID is preferred
  });


  // ****************** Links (Hierarchy) ***************************
  const link = g.selectAll('path.link')
      .data(links, d => d.target.id);

  const linkEnter = link.enter().insert('path', "g")
      .attr("class", "link")
      .attr('d', d => {
        const o = {x: source.x0, y: source.y0};
        return diagonal(o, o);
      });

  const linkUpdate = linkEnter.merge(link);

  linkUpdate.transition()
      .duration(duration)
      .attr('d', d => diagonal(d.source, d.target));

  link.exit().transition()
      .duration(duration)
      .attr('d', d => {
        const o = {x: source.x, y: source.y};
        return diagonal(o, o);
      })
      .remove();

  // ****************** Dependency Links (Explicit) ***************************
  // Calculate active dependencies based on current visible nodes
  let depLinksData = [];
  if (showDependencies && parsedData.dependencies) {
      parsedData.dependencies.forEach(dep => {
          const sourceNode = nodeMap.get(dep.source);
          const targetNode = nodeMap.get(dep.target);
          
          if (sourceNode && targetNode) {
              depLinksData.push({source: sourceNode, target: targetNode});
          }
      });
  }

  const depLink = g.selectAll('path.dep-link')
      .data(depLinksData, d => d.source.id + "-" + d.target.id);

  const depLinkEnter = depLink.enter().append('path')
      .attr("class", "dep-link")
      .attr('d', d => {
           // Start from wherever the source is currently (animation) -> usually nice to just fade in or pop in
           // For simplicity, we calculate the curve immediately or use the source position
           return dependencyPath(d.source, d.target);
      })
      .style("opacity", 0);

  depLinkEnter.transition().duration(duration).style("opacity", 0.6);

  depLink.transition().duration(duration)
      .attr('d', d => dependencyPath(d.source, d.target))
      .style("opacity", 0.6);

  depLink.exit().transition().duration(duration).style("opacity", 0).remove();


  // ****************** Nodes ***************************
  const node = g.selectAll('g.node')
      .data(nodes, d => d.id || (d.id = ++i));

  const nodeEnter = node.enter().append('g')
      .attr('class', 'node')
      .attr("transform", d => "translate(" + source.y0 + "," + source.x0 + ")")
      .on('click', click);

  nodeEnter.append('circle')
      .attr('r', 1e-6)
      .attr('class', d => {
          const status = (d.data.metadata && d.data.metadata.status) || 'default';
          return `status-${status.replace(' ', '-')}`;
      });

  nodeEnter.append('text')
      .attr("dy", ".35em")
      .attr("x", d => d.children || d._children ? -13 : 13)
      .attr("text-anchor", d => d.children || d._children ? "end" : "start")
      .text(d => {
          let title = d.data.title;
          return title.length > 30 ? title.substring(0, 30) + '...' : title;
      })
      .style('fill-opacity', 1e-6);

  const nodeUpdate = nodeEnter.merge(node);

  nodeUpdate.transition()
      .duration(duration)
      .attr("transform", d => "translate(" + d.y + "," + d.x + ")");

  nodeUpdate.select('circle')
      .attr('r', 8)
      .style("fill", d => d._children ? "#fff" : "") 
      .attr('class', d => `status-${((d.data.metadata && d.data.metadata.status) || 'default').replace(' ', '-')}`);

  nodeUpdate.select('text').style("fill-opacity", 1);

  const nodeExit = node.exit().transition()
      .duration(duration)
      .attr("transform", d => "translate(" + source.y + "," + source.x + ")")
      .remove();

  nodeExit.select('circle').attr('r', 1e-6);
  nodeExit.select('text').style('fill-opacity', 1e-6);


  nodes.forEach(d => {
    d.x0 = d.x;
    d.y0 = d.y;
  });

  function diagonal(s, d) {
    return `M ${s.y} ${s.x}
            C ${(s.y + d.y) / 2} ${s.x},
              ${(s.y + d.y) / 2} ${d.x},
              ${d.y} ${d.x}`;
  }

  function dependencyPath(s, t) {
      // Custom path for dependencies - larger arc to avoid hierarchy lines?
      // Or just a straightish Bezier
      const dx = t.y - s.y;
      const dy = t.x - s.x;
      const dr = Math.sqrt(dx * dx + dy * dy) * 1.5; // Controls curvature
      
      // Arc path
      return `M${s.y},${s.x}A${dr},${dr} 0 0,1 ${t.y},${t.x}`;
  }

  function click(event, d) {
    showDetails(d.data);
    if (d.children) {
        d._children = d.children;
        d.children = null;
    } else {
        d.children = d._children;
        d._children = null;
    }
    update(d);
  }
}

function showDetails(data) {
    const sidebar = document.getElementById('sidebar');
    const container = document.getElementById('details');
    sidebar.style.display = 'block';

    const status = (data.metadata.status || 'todo').replace(' ', '-');
    
    let metaHtml = '';
    for (const [key, value] of Object.entries(data.metadata)) {
        if (key === 'status') continue;
        metaHtml += `<div><strong>${key}:</strong> ${JSON.stringify(value).replace(/"/g, '')}</div>`;
    }

    container.innerHTML = `
        <span class="meta-tag tag-${status}">${status.toUpperCase().replace('-', ' ')}</span>
        <h2>${data.title}</h2>
        <div style="margin-bottom: 20px; font-size: 0.9em; color: #7f8c8d;">
            ${metaHtml}
        </div>
        <hr style="border: 0; border-top: 1px solid #eee;"/>
        <div class="content-block">
            <pre style="background:none; padding:0; white-space: pre-wrap; font-family: inherit;">${data.content || "No content."}</pre>
        </div>
    `;
}
</script>
</body>
</html>
"""

def node_to_dict(node):
    return {
        "title": node.title,
        "metadata": node.metadata,
        "content": node.content,
        "children": [node_to_dict(child) for child in node.children]
    }

def collect_dependencies(node, dependencies_list=None):
    if dependencies_list is None:
        dependencies_list = []
    
    # Check for blocked_by
    if node.metadata and 'blocked_by' in node.metadata and 'id' in node.metadata:
        target_id = node.metadata['id']
        blockers = node.metadata['blocked_by']
        if isinstance(blockers, list):
            for source_id in blockers:
                 dependencies_list.append({"source": source_id, "target": target_id})
    
    for child in node.children:
        collect_dependencies(child, dependencies_list)
        
    return dependencies_list

def generate_html(target_file):
    print("Checking dependencies...")
    ensure_d3()

    print(f"Parsing {target_file}...")
    parser = MarkdownParser()
    root_node = parser.parse_file(target_file)

    # Adjust root logic - we want the real content root
    if root_node.title == "Root" and len(root_node.children) == 1:
        tree_root = root_node.children[0]
    elif root_node.title == "Root" and len(root_node.children) > 1:
        tree_root = root_node
        tree_root.title = "Central Planner" # Rename for viz
    else:
        tree_root = root_node
        
    tree_data = node_to_dict(tree_root)
    dependencies = collect_dependencies(tree_root)
    
    full_data = {
        "tree": tree_data,
        "dependencies": dependencies
    }
        
    json_str = json.dumps(full_data)
    b64_data = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
    
    html_content = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", b64_data)
    return html_content

def main():
    target_file = os.path.join(planner_dir, "MASTER_PLAN.md")
    if not os.path.exists(target_file):
        print("MASTER_PLAN.md not found")
        sys.exit(1)

    html_content = generate_html(target_file)
    
    output_path = os.path.join(planner_dir, "master_plan_interactive.html")
    with open(output_path, 'w') as f:
        f.write(html_content)
        
    print(f"Visualization saved to: {output_path}")
    
    try:
        webbrowser.open('file://' + os.path.abspath(output_path))
    except Exception as e:
        print(f"Could not open browser: {e}")

if __name__ == "__main__":
    main()
