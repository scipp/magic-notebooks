import inspect
import networkx as nx
import matplotlib.pyplot as plt

class ParameterResolver:
    def __init__(self, functions, params, aliases=None):
        self.functions = functions
        self.params = params
        self.aliases = aliases or {}
        self._stack = set()
        self.last_inputs = {}  # for partial recomputation

        # Build reverse alias map
        self.reverse_aliases = {}
        for alias, canonical in self.aliases.items():
            self.reverse_aliases.setdefault(canonical, []).append(alias)

    def canonical(self, name):
        return self.aliases.get(name, name)

    def resolve(self, requested, force_recompute=False, verbose=False):
        results = {}
        for name in requested:
            canonical_name = self.canonical(name)
            results[name] = self._resolve_one(canonical_name, force_recompute, verbose)
        return results

    def _resolve_one(self, name, force_recompute, verbose):
        # Check existing values (including aliases)
        if not force_recompute:
            val = self._get_param(name)
            if val is not None:
                return val

        if name in self._stack:
            raise RuntimeError(f"Cyclic dependency detected for '{name}'")

        self._stack.add(name)

        func, outputs = self._find_function_for(name)
        if func is None:
            val = self._get_param(name)
            if val is not None:
                return val
            raise KeyError(f"No function found to compute '{name}'")

        if verbose:
            print(f"Computing {name} using {func.__name__}")

        # sig = inspect.signature(func)
        # input_names = list(sig.parameters.keys())

        # inputs = {
        #     p: self._resolve_one(self.canonical(p), force_recompute, verbose)
        #     for p in input_names
        # }

        sig = inspect.signature(func)
        inputs = {}

        for p_name, p in sig.parameters.items():
            canonical_p = self.canonical(p_name)

            # 1. If parameter exists in params (or alias), use it
            val = self._get_param(canonical_p)
            if val is not None:
                inputs[p_name] = val
                continue

            # 2. If parameter can be computed by another function → compute it
            subfunc, _ = self._find_function_for(canonical_p)
            if subfunc is not None:
                inputs[p_name] = self._resolve_one(canonical_p, force_recompute, verbose)
                continue

            # 3. Otherwise, use default value if available
            if p.default is not inspect._empty:
                inputs[p_name] = p.default
                continue

            # 4. If no default and no function → error
            raise KeyError(f"Cannot resolve parameter '{p_name}' for function '{func.__name__}'")
        

        # Partial recomputation logic
        if not force_recompute:
            if not self.should_recompute(func.__name__, inputs):
                if verbose:
                    print(f"Skipping recomputation of {name}, inputs unchanged")
                self._stack.remove(name)
                return self._get_param(name)

        # Run function
        result = func(**inputs)
        self.last_inputs[func.__name__] = inputs.copy()

        # Store outputs
        if isinstance(outputs, tuple):
            for key, val in zip(outputs, result):
                self._store_param(key, val)
        else:
            self._store_param(outputs, result)

        self._stack.remove(name)
        return self._get_param(name)

    def should_recompute(self, func_name, current_inputs):
        last = self.last_inputs.get(func_name)
        if last is None:
            return True
        for key, val in current_inputs.items():
            if key not in last or last[key] != val:
                return True
        return False

    def _find_function_for(self, name):
        for outputs, func in self.functions.items():
            if outputs == name:
                return func, outputs
            if isinstance(outputs, tuple) and name in outputs:
                return func, outputs
        return None, None

    def _get_param(self, name):
        if name in self.params:
            return self.params[name]
        for alias in self.reverse_aliases.get(name, []):
            if alias in self.params:
                return self.params[alias]
        return None

    def _store_param(self, name, value):
        self.params[name] = value
        for alias in self.reverse_aliases.get(name, []):
            self.params[alias] = value

    # Dependency graph
    def build_dependency_graph(self):
        graph = {}
        for outputs, func in self.functions.items():
            sig = inspect.signature(func)
            inputs = list(sig.parameters.keys())
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            for inp in inputs:
                inp = self.canonical(inp)
                graph.setdefault(inp, [])
                for out in outputs:
                    graph[inp].append(out)
        return graph

    def visualize_dependency_graph(self):
        graph = self.build_dependency_graph()
        G = nx.DiGraph()
        for inp, outs in graph.items():
            for out in outs:
                G.add_edge(inp, out)

        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=2000,
                node_color="lightblue", arrowsize=20, font_size=12)
        plt.title("Parameter Dependency Graph")
        plt.show()


if __name__ == '__main__':
    def func_a(x, y):
        return x + y

    def func_bc(a, t: int = 0):
        return a * 2 + t, a * 3 + t

    functions = {
        "a": func_a,
        ("b", "c"): func_bc,
    }

    params = {
        "x": 1,
        "y": 2,
        "a_new": 100,   # alias for "a"
    }

    aliases = {
        "a_new": "a"
    }

    resolver = ParameterResolver(functions, params, aliases)

    print("force_recompute=False")
    print(resolver.resolve(("b",), force_recompute=False, verbose=True))

    print("force_recompute=True")
    print(resolver.resolve(("b",), force_recompute=True, verbose=True))
    resolver.visualize_dependency_graph()