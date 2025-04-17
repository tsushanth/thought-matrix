class ThoughtMatrix:
    def __init__(self, model, n_partitions=3):
        self.model = model
        self.n = n_partitions
        self.mind_map = np.zeros((n_partitions, n_partitions))
        self.layer_assignments = self._partition_model()
        
    def _partition_model(self):
        """Split model into NxN logical blocks"""
        layers = list(model.children()) if hasattr(model, 'children') else []
        partitions = np.array_split(layers, self.n**2)
        return partitions.reshape((self.n, self.n))

    def infer_with_tracing(self, prompt):
        """Run inference while tracking activation paths"""
        hooks = self._attach_probes()
        output = self.model.generate(prompt)
        self._detach_probes(hooks)
        return output, self.mind_map

    def _attach_probes(self):
        """Attach forward hooks to track activation"""
        hooks = []
        for i in range(self.n):
            for j in range(self.n):
                def hook(module, inp, out, i=i, j=j):
                    self.mind_map[i,j] += 1  # Track activation
                hooks.append(self.layer_assignments[i,j].register_forward_hook(hook))
        return hooks

class ThoughtEvaluator:
    def __init__(self, eval_model):
        self.eval_model = eval_model
        
    def evaluate_thought(self, mind_map, prompt, output, 
                        criteria=["creativity", "accuracy"]):
        analysis_prompt = f"""
        Analyze this LLM thought pattern:
        Prompt: {prompt}
        Output: {output}
        Activation Map:\n{mind_map}
        
        Rate 1-10 on: {','.join(criteria)}
        """
        return self.eval_model.generate(analysis_prompt)