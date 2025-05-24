# build_network.py
import re
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import itertools
import math
from pathlib import Path

# --- Constants for WoS Field Codes ---
# (Keep the constants definitions as provided before)
FN_FIELD = 'FN'; VR_FIELD = 'VR'; PT_FIELD = 'PT'; AU_FIELD = 'AU'; AF_FIELD = 'AF'
TI_FIELD = 'TI'; SO_FIELD = 'SO'; LA_FIELD = 'LA'; DT_FIELD = 'DT'; DE_FIELD = 'DE'
ID_FIELD = 'ID'; AB_FIELD = 'AB'; C1_FIELD = 'C1'; EM_FIELD = 'EM'; OI_FIELD = 'OI'
CR_FIELD = 'CR'; NR_FIELD = 'NR'; TC_FIELD = 'TC'; PY_FIELD = 'PY'; VL_FIELD = 'VL'
IS_FIELD = 'IS'; BP_FIELD = 'BP'; EP_FIELD = 'EP'; PG_FIELD = 'PG'; UT_FIELD = 'UT'
ER_FIELD = 'ER'; EF_FIELD = 'EF'

# --- Function: parse_wos_file ---
# (Keep the parse_wos_file function exactly as provided before)
def parse_wos_file(filepath):
    """
    Parses a Web of Science plain text file, correctly handling multi-line fields.
    """
    publications = []
    current_pub = {}
    current_field = None
    line_num = 0 # Initialize line number for error reporting

    try:
        # --- File reading logic (same as before) ---
        encodings_to_try = ['utf-8', 'latin-1']
        file_content = None
        for enc in encodings_to_try:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    file_content = f.readlines()
                print(f"  Successfully read file {Path(filepath).name} with encoding: {enc}")
                break
            except UnicodeDecodeError:
                continue
        if file_content is None:
            print(f"Warning: Could not decode file {filepath} with attempted encodings. Skipping.")
            return []
        # --- End file reading logic ---

        for line_num, line in enumerate(file_content):
            original_line = line # Keep original line to check indentation
            line = line.strip() # Work with the stripped version

            if not line: continue # Skip empty lines

            # Check if line starts with a 2-character field code + space
            field_code = None
            value = None
            is_continuation = False

            if len(line) > 3 and line[2] == ' ' and line[:2].isalnum() and line[:2].isupper():
                 field_code = line[:2]
                 value = line[3:].strip()
            else:
                 # Does not start with a field code, treat as continuation or special line (like ER/FN/VR)
                 is_continuation = True
                 value = line # Use the stripped line content as value

            # --- Handle End of Record ---
            # Check the START of the original line for ER, FN, VR which don't have standard spacing
            # Or check the stripped line if it's just ER
            if original_line.startswith(ER_FIELD) or line == ER_FIELD:
                if current_pub: # If we have data for a publication
                    # Add UT if missing before saving
                    if UT_FIELD not in current_pub:
                        current_pub[UT_FIELD] = f"MISSING_UT_{len(publications)}"
                    publications.append(current_pub)
                current_pub = {} # Reset for the next publication
                current_field = None # Reset current field context
                continue # Move to the next line after processing ER

            # Handle special start-of-file tags if necessary (e.g., FN, VR)
            if original_line.startswith(FN_FIELD) or original_line.startswith(VR_FIELD):
                 # These might signal the start, reset context if needed, but usually handled by ER
                 # We can parse them like normal fields if we hit them
                 if line[2] == ' ': # Check format just in case
                     field_code = line[:2]
                     value = line[3:].strip()
                     is_continuation = False # Treat as a new field
                 else: # If format is weird, skip or log?
                      continue


            # --- Process based on whether it's a new field or continuation ---
            if not is_continuation and field_code:
                # === It's a new field ===
                current_field = field_code # Update the current field context

                # Fields where each line (starting with code OR indented) is a SEPARATE item
                if current_field in [CR_FIELD, AU_FIELD, AF_FIELD, C1_FIELD, DE_FIELD, ID_FIELD]:
                    if current_field not in current_pub:
                        current_pub[current_field] = []
                    current_pub[current_field].append(value) # Add value as a new list item

                # Fields where continuation lines should be appended to the value (e.g., Abstract)
                elif current_field in [AB_FIELD, TI_FIELD]:
                    current_pub[current_field] = value # Initialize the string value

                # Other single-value fields
                else:
                    current_pub[current_field] = value # Assign the value directly

            elif is_continuation and current_field:
                # === It's a continuation line ===
                # Check if it's for a list-based field and properly indented (WoS uses 3 spaces)
                if current_field in [CR_FIELD, AU_FIELD, AF_FIELD, C1_FIELD, DE_FIELD, ID_FIELD]:
                    if original_line.startswith('   ') and current_field in current_pub:
                        # Indented line for list field -> Add as a NEW item
                        current_pub[current_field].append(value)
                    elif current_field in current_pub and current_pub[current_field]:
                        # Not indented or list not started? Append to the *last* item (handles wrapped text within one entry)
                        current_pub[current_field][-1] += " " + value
                    # else: List not initialized, ignore continuation? Or log warning?

                # Check if it's for a string-based field
                elif current_field in [AB_FIELD, TI_FIELD]:
                    if current_field in current_pub:
                        current_pub[current_field] += " " + value # Append to string
                    # else: Field not initialized, ignore continuation?

                # Else: Continuation for a field we don't explicitly handle list/string append for (ignore?)

            # Note: We don't need the specific 'EF' (End of File) check usually, ER handles records.

        # Add the last publication if the file doesn't end neatly with ER
        if current_pub and UT_FIELD in current_pub:
             publications.append(current_pub)

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e:
        print(f"An error occurred during parsing file {filepath}: {e} near line {line_num + 1}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return [] # Return empty or partially parsed data

    return publications

# --- Function: normalize_cited_ref ---
# (Keep the normalize_cited_ref function exactly as provided before)
def normalize_cited_ref(ref_string):
    """
    Attempts to parse and normalize a cited reference string (more aggressively).
    Returns a standardized string identifier or None if parsing fails.
    Example target format: "AUTHOR, YEAR, SOURCE" (all uppercase, simplified)
    """
    ref_string = ref_string.strip()
    # Handle cases like "[Anonymous], INT J BEHAV NUTR PHY" where year is missing early
    # Try to extract year first from anywhere in the string
    year = None
    year_match = re.search(r'\b(1[89]\d{2}|20\d{2})\b', ref_string) # Search whole string
    if year_match:
        year = year_match.group(1)
    else:
        year = "UNKNOWN_YEAR" # Fallback if no year found anywhere

    # Split by comma AFTER potentially finding year
    parts = [p.strip() for p in ref_string.split(',')]
    if not parts: return None

    # 1. Author Normalization (Uppercase, remove periods)
    raw_author = parts[0].upper()
    if raw_author.startswith('[ANONYMOUS]'):
        author = "ANONYMOUS"
    else:
        # Remove periods and excessive spaces
        author = re.sub(r'\.', '', raw_author).strip()
        author = re.sub(r'\s+', ' ', author)
        # Let's NOT truncate initials for now, just clean name: "DAVIS F D" -> "DAVIS F D"
        # More complex initial handling might be needed but adds risk

    # 2. Source Normalization (Uppercase, remove Vol/Page/DOI info, basic cleaning)
    source = "UNKNOWN_SOURCE" # Default
    potential_source_str = ""
    # Heuristic: Source usually starts after Author and potentially Year if year was in parts[1]
    if len(parts) > 1:
        if parts[1] == year: # If year was the second part
             potential_source_str = ", ".join(parts[2:]).upper()
        else: # Year was found elsewhere or missing, source likely starts from part 2
             potential_source_str = ", ".join(parts[1:]).upper()

    if potential_source_str:
        # Remove details like V..., P..., DOI..., HTTP... etc. more broadly
        # This regex looks for ", V" or ", P" or ", DOI" etc. and removes everything after
        cleaned_source = re.split(r',\s+(?:V|P|DOI|HTTP|WWW)\b', potential_source_str, 1)[0].strip()

        # Further cleanups
        cleaned_source = cleaned_source.strip(', ') # Remove leading/trailing commas/spaces
        # Special case: if the source is just the year, mark as unknown
        if cleaned_source == year:
             source = "UNKNOWN_SOURCE"
        elif cleaned_source:
             source = re.sub(r'\s+', ' ', cleaned_source) # Consolidate whitespace
             # Truncate source to avoid excessive detail/variation? (optional)
             # max_source_len = 40
             # source = source[:max_source_len]

    # Return standardized key: AUTHOR, YEAR, SOURCE
    # Only return if author and year are valid (not None or empty strings)
    if author and year:
        return f"{author}, {year}, {source}"
    else:
        # Return None if essential parts like author couldn't be determined
        # This prevents creating nodes with invalid keys.
        return None


# --- Function: build_cocitation_network ---
def build_cocitation_network(publications):
    """ Builds a co-citation network from parsed WoS publications (with debugging). """
    G = nx.Graph()
    cited_ref_counts = Counter()
    cocitation_links = defaultdict(int)
    node_info = {} # Store info like year for each node

    print(f"Building network from {len(publications)} citing publications...")
    processed_pubs = 0
    debug_prints_done = 0 # Counter for how many publications we print debug info for

    for pub in publications:
        citing_pub_id = pub.get(UT_FIELD, f'UnknownUT_{processed_pubs}') # Ensure unique unknown ID
        citing_pub_year = pub.get(PY_FIELD, None) # Year the co-citation happened
        cited_refs_raw = pub.get(CR_FIELD, [])

        normalized_refs = []
        # --- Debug: Print raw CRs for the first few pubs ---
        if debug_prints_done < 3 and cited_refs_raw: # Only print if there are references
             print(f"\n--- Debug: Citing Pub ID: {citing_pub_id} ---")
             print(f"Raw CRs ({len(cited_refs_raw)}):")
             # Limit printing raw refs if list is very long
             max_raw_to_print = 20
             for i, r in enumerate(cited_refs_raw[:max_raw_to_print]):
                 print(f"  [{i}] {r}")
             if len(cited_refs_raw) > max_raw_to_print:
                 print(f"  ... (and {len(cited_refs_raw) - max_raw_to_print} more)")

        for ref_str in cited_refs_raw:
            # *** Make sure you are using the intended normalize_cited_ref function here ***
            norm_ref = normalize_cited_ref(ref_str) # Call the (potentially revised) function
            if norm_ref:
                normalized_refs.append(norm_ref)
                # Count total citations for each ref
                cited_ref_counts[norm_ref] += 1
                # Store year if possible (basic extraction from normalized string)
                if norm_ref not in node_info:
                     parts = norm_ref.split(', ')
                     ref_year = parts[1] if len(parts) > 1 and parts[1].isdigit() else None
                     node_info[norm_ref] = {'year': ref_year, 'label': norm_ref}

        # --- Debug: Print normalized refs and pairs for the first few pubs ---
        if debug_prints_done < 3 and normalized_refs:
            print(f"Normalized Refs ({len(normalized_refs)}):")
            # Limit printing normalized refs if list is very long
            max_norm_to_print = 20
            for i, r in enumerate(normalized_refs[:max_norm_to_print]):
                 print(f"  [{i}] {r}")
            if len(normalized_refs) > max_norm_to_print:
                 print(f"  ... (and {len(normalized_refs) - max_norm_to_print} more)")
            print("Generating combinations...")

        # Generate co-citation pairs (edges)
        pair_count_this_pub = 0
        # Use itertools.combinations to get all unique pairs WITHIN the normalized list for THIS publication
        if len(normalized_refs) >= 2: # Optimization: Only run combinations if there are at least 2 refs
            for ref1, ref2 in itertools.combinations(normalized_refs, 2):
                # Ensure consistent edge ordering (important for defaultdict)
                edge = tuple(sorted((ref1, ref2)))
                cocitation_links[edge] += 1
                pair_count_this_pub += 1
        else:
             # --- Debug: Indicate if not enough refs for pairs ---
             if debug_prints_done < 3 and normalized_refs:
                 print("Not enough normalized references (need >= 2) to form pairs for this publication.")


        # --- Debug: Print number of pairs found ---
        if debug_prints_done < 3 and cited_refs_raw: # Only increment if we printed debug info
             print(f"Found {pair_count_this_pub} co-citation pairs for this publication.")
             debug_prints_done += 1 # Increment after processing one pub fully

        processed_pubs += 1
        if processed_pubs % 500 == 0: # Print progress less often
             print(f"  Processed {processed_pubs}/{len(publications)} publications...")

    print(f"\nFinished processing publications.")
    print(f"Total unique cited references identified (nodes): {len(cited_ref_counts)}")
    print(f"Total unique co-citation links found (edge types): {len(cocitation_links)}")
    # Calculate total co-citations (sum of weights)
    total_cocitations = sum(cocitation_links.values())
    print(f"Total co-citation instances (sum of edge weights): {total_cocitations}")


    # Add nodes to the graph
    print("Adding nodes to graph...")
    for ref, count in cited_ref_counts.items():
        info = node_info.get(ref, {})
        G.add_node(
            ref,
            label=info.get('label', ref), # Use label or ref itself
            freq=count,
            year=info.get('year', None) # Add year if found
        )

    # Add edges to the graph
    print("Adding edges to graph...")
    edges_added = 0
    for (ref1, ref2), weight in cocitation_links.items():
        # Ensure nodes exist before adding edge (should always be true with this logic)
        if G.has_node(ref1) and G.has_node(ref2):
            G.add_edge(ref1, ref2, weight=float(weight)) # Use float for weight like graphml
            edges_added += 1

    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges (Edges added based on {edges_added} unique links).")
    return G

# --- Function: save_graph_to_graphml ---
# Added function to handle saving with correct attributes and types
def save_graph_to_graphml(graph, filepath):
    """Saves the networkx graph to GraphML with essential attributes only."""
    if graph.number_of_nodes() == 0:
        print("Graph is empty, skipping save.")
        return

    print(f"Preparing graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges for saving...")

    # Create a copy to modify attributes for saving without changing the original graph object
    graph_to_save = graph.copy()

    # --- Keep only essential attributes ---
    # Node attributes to keep: label, freq, year
    # Edge attributes to keep: weight
    nodes_to_remove_attrs = []
    for node, data in graph_to_save.nodes(data=True):
        attrs_to_remove = [attr for attr in data if attr not in ['label', 'freq', 'year']]
        if attrs_to_remove:
            nodes_to_remove_attrs.append((node, attrs_to_remove))
        # Ensure essential attributes have correct types
        data['label'] = str(data.get('label', node))
        data['freq'] = int(data.get('freq', 0))
        data['year'] = str(data.get('year', '')) # Keep year as string for GraphML consistency

    # Remove non-essential attributes (modifying the copy)
    for node, attrs in nodes_to_remove_attrs:
        for attr in attrs:
            del graph_to_save.nodes[node][attr]

    edges_to_remove_attrs = []
    for u, v, data in graph_to_save.edges(data=True):
        attrs_to_remove = [attr for attr in data if attr not in ['weight']]
        if attrs_to_remove:
            edges_to_remove_attrs.append(((u, v), attrs_to_remove))
        # Ensure essential attribute has correct type
        data['weight'] = float(data.get('weight', 0.0))

    for edge, attrs in edges_to_remove_attrs:
        for attr in attrs:
            del graph_to_save.edges[edge][attr]


    # --- Saving Logic ---
    try:
        output_dir = Path(filepath).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define keys for the essential attributes being saved
        keys_node = {"label": "string", "freq": "integer", "year": "string"}
        keys_edge = {"weight": "float"}

        # Write GraphML using lxml (preferred)
        # Use explicit attribute types for clarity, though infer_numeric_types helps
        nx.write_graphml_lxml(graph_to_save, str(filepath),
                              node_attr_types=keys_node, # Pass the types for saved attributes
                              edge_attr_types=keys_edge, # Pass the types for saved attributes
                              named_key_ids=True,
                              infer_numeric_types=False) # Set to False as we provide types explicitly
        print(f"Graph successfully saved to {filepath}")

    except ImportError:
        print("Error: Saving to GraphML requires 'lxml'. Trying basic XML writer.")
        try:
             # Basic writer might not handle explicit types well
            nx.write_graphml_xml(graph_to_save, str(filepath), named_key_ids=True)
            print(f"Graph saved using basic XML writer (lxml preferred).")
        except Exception as e_xml:
            print(f"Error saving graph with basic XML writer: {e_xml}")
    except TypeError as te:
         # Catch potential TypeError if write_graphml_lxml signature changed again
         if 'node_attr_types' in str(te) or 'edge_attr_types' in str(te):
             print("Warning: 'node_attr_types'/'edge_attr_types' might not be supported by nx.write_graphml_lxml.")
             print("Attempting save without explicit types...")
             try:
                 nx.write_graphml_lxml(graph_to_save, str(filepath), named_key_ids=True, infer_numeric_types=True)
                 print(f"Graph successfully saved to {filepath} (types inferred).")
             except Exception as e_infer:
                  print(f"Error saving graph even with inferred types: {e_infer}")
         else:
              print(f"An unexpected TypeError occurred during saving: {te}")
    except Exception as e:
        print(f"An error occurred saving the graph: {e}")


# --- Main Execution Logic ---
if __name__ == "__main__":
    # --- Define paths (same as before) ---
    script_dir = Path(__file__).parent
    wos_data_dir = script_dir / 'data' / 'wos'
    graphml_output_dir = script_dir / 'data' / 'graphml'
    output_filename = 'temp_output_graph.graphml' # New filename for pruned graph
    output_filepath = graphml_output_dir / output_filename

    print(f"Looking for WoS files matching 'savedrecs*.txt' in: {wos_data_dir}")

    all_publications = []
    # --- File finding/parsing loop (same as before) ---
    if not wos_data_dir.is_dir():
        print(f"Error: Input directory not found: {wos_data_dir}")
    else:
        # Find and sort files (logic from previous step)
        all_txt_files = list(wos_data_dir.glob('*.txt'))
        file_pattern = re.compile(r"savedrecs(?: \((\d+)\))?\.txt")
        files_with_num = []
        for f_path in all_txt_files:
            match = file_pattern.match(f_path.name)
            if match:
                num = int(match.group(1)) if match.group(1) else 0
                files_with_num.append((num, f_path))
        files_with_num.sort(key=lambda x: x[0])
        wos_files_to_process = [f_path for num, f_path in files_with_num]

        print(f"Found {len(wos_files_to_process)} files matching the pattern to process.")

        if not wos_files_to_process:
            print("No files matching the 'savedrecs*.txt' pattern found.")
        else:
            # Parse files
            for wos_file in wos_files_to_process:
                print(f"Parsing {wos_file.name}...")
                pubs = parse_wos_file(wos_file)
                all_publications.extend(pubs)
                print(f"  Added {len(pubs)} publications. Total: {len(all_publications)}")

            if not all_publications:
                print("No publications were parsed successfully.")
            else:
                # --- Build the full network ---
                G_full = build_cocitation_network(all_publications)
                print(f"\n--- Full Network Stats ---")
                print(f"Nodes: {G_full.number_of_nodes()}, Edges: {G_full.number_of_edges()}")

                if G_full.number_of_nodes() > 0 and G_full.number_of_edges() > 0:
                    # --- PRUNING STEPS ---
                    print("\n--- Pruning Network ---")

                    # 1. Remove low-weight edges (e.g., weight <= 1)
                    min_weight = 1 # Adjust this threshold if needed (e.g., 2 to keep only stronger links)
                    print(f"Removing edges with weight <= {min_weight}...")
                    edges_to_remove = [(u, v) for u, v, data in G_full.edges(data=True) if data.get('weight', 0) <= min_weight]
                    G_pruned_edges = G_full.copy() # Work on a copy
                    G_pruned_edges.remove_edges_from(edges_to_remove)
                    print(f"Edges after weight pruning: {G_pruned_edges.number_of_edges()}")

                    # 2. Remove isolated nodes (nodes with no edges after edge pruning)
                    G_pruned_edges.remove_nodes_from(list(nx.isolates(G_pruned_edges)))
                    print(f"Nodes after removing isolates: {G_pruned_edges.number_of_nodes()}")

                    # 3. Keep only the largest connected component (LCC)
                    # Ensure the graph is treated as undirected for connected components
                    if G_pruned_edges.number_of_nodes() > 0:
                         print("Finding connected components...")
                         # Use the graph copy for component finding
                         graph_for_components = G_pruned_edges.to_undirected() if nx.is_directed(G_pruned_edges) else G_pruned_edges

                         components = list(nx.connected_components(graph_for_components))
                         if components:
                             largest_component_nodes = max(components, key=len)
                             G_pruned = G_pruned_edges.subgraph(largest_component_nodes).copy() # Create subgraph and copy it
                             print(f"Kept largest connected component.")
                             print(f"--- Final Pruned Network Stats ---")
                             print(f"Nodes: {G_pruned.number_of_nodes()}, Edges: {G_pruned.number_of_edges()}")
                         else:
                             print("No connected components found after pruning.")
                             G_pruned = G_pruned_edges # Use the edge-pruned graph
                    else:
                         print("Graph is empty after edge pruning.")
                         G_pruned = G_pruned_edges # Use the empty graph

                    # --- Save the PRUNED graph ---
                    # Use the revised save function with reduced attributes
                    save_graph_to_graphml(G_pruned, output_filepath)

                else:
                     print("Full graph has no edges or nodes, skipping pruning and saving.")