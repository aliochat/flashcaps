import torch
import torch.nn as nn
import torch.functional as F


class BeamSearchScorer(nn.Module):
    def __init__(
        self,
        beam_size: int,
        batch_size: int,
        max_length: int,
        length_penalty: float,
        early_stopping: bool,
        num_hypo_to_return: int,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        device: str,
    ):
        """
        BeamSearchScorer class for managing beam search hypotheses.

        Parameters:
        -----------
        beam_size : int
            Number of beams to keep in the search.

        batch_size : int
            Size of the input batch.

        max_length : int
            Maximum length of the generated hypotheses.

        length_penalty : float
            Length penalty for hypotheses scoring.

        early_stopping : bool
            Whether to stop generation early if any beam is finished.

        num_hypo_to_return : int
            Number of hypotheses to return.

        bos_token_id : int
            ID of the beginning-of-sequence token.

        eos_token_id : int
            ID of the end-of-sequence token.

        pad_token_id : int
            ID of the padding token.

        device : str
            Device to run computations on.
        """
        super().__init__()
        self.beam_size = beam_size
        self.batch_size = batch_size

        self.max_length = max_length
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_hypo_to_return = num_hypo_to_return

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self._done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self._beam_hyps_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._beam_hyps_worst_scores = torch.zeros(batch_size, device=device) + 1e9
        self._beam_hyps = []
        self._beam_scores = []

    def init(self):
        """
        Reinitialize the beam search state.
        """
        self._done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self._beam_hyps_count = torch.zeros(
            self.batch_size, dtype=torch.long, device=self.device
        )
        self._beam_hyps_worst_scores = (
            torch.zeros(self.batch_size, device=self.device) + 1e9
        )
        self._beam_hyps = []
        self._beam_scores = []

    def is_done(self) -> torch.Tensor:
        """
        Check if all batches are finished.

        Returns:
        --------
            torch.Tensor: True if all batches are finished, else False.
        """
        return self._done.all()

    def hypo_len(self, batch_idx: int) -> int:
        """
        Get the number of hypotheses for a certain batch.

        Parameters:
        -----------
            batch_idx : int
                Index of the batch.

        Returns:
        --------
            int: Number of hypotheses for the batch.
        """
        return self._beam_hyps_count[batch_idx]

    def hypo_add(self, hyp: torch.Tensor, sum_logprobs: float, batch_idx: int):
        """
        Given a new hypothesis (hyp), its associated sum of log probabilities (sum_logprobs), and the index of the batch (batch_idx), 
        the method calculates a score for the hypothesis based on the provided log probabilities. This score is adjusted with a length penalty factor. 
        The method then determines whether the new hypothesis should be added to the list based on two conditions: 
        - batch's current count of hypotheses is less than the specified beam size, or 
        - score of the new hypothesis is higher than the worst score among the current hypotheses for that batch. 
        
        If either condition is met, the method inserts the hypothesis and its score into their respective lists, 
        maintaining the order according to the score. If the number of hypotheses exceeds the beam size, the worst hypothesis is pruned from the list, 
        and the worst score is updated. If there is still room for additional hypotheses, the method updates the worst score and the count of hypotheses for the batch. 
        This process ensures that the hypotheses with the highest scores and the most potential for improvement are retained in the ongoing search.

        Parameters:
        -----------
        hyp : torch.Tensor)
            New hypothesis containing the generated tokens so far for the hypothesis.

        sum_logprobs : float
            Sum of log probabilities over the hypothesis.

        batch_idx : int
            Index of the batch.
        """
        # Score of the hypothesis is the sum of log probabilities over it's length
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)

        # Number of finished hypotheses for the batch
        hyps_count = self.hypo_len(batch_idx)

        # If we don't have enough hypothesis or the hyp's score is better than the worst score in the list
        if (
            hyps_count < self.beam_size
            or score > self._beam_hyps_worst_scores[batch_idx]
        ):
            # Find the index of the first slot where we can insert the the new hypothesis for the batch
            beam_idx = (
                torch.sum(self._beam_hyps_count[:batch_idx])
                if batch_idx != 0
                else torch.tensor(0, dtype=torch.long)
            )
            self._beam_scores.insert(beam_idx, torch.tensor([score]))
            self._beam_hyps.insert(beam_idx, hyp)

            # If the number of hypotheses for the batch is greater than the beam size, remove the worst hypothesis
            if hyps_count + 1 > self.beam_size:
                sorted_next_scores, sorted_indices = torch.topk(
                    torch.cat(self._beam_scores)[beam_idx : beam_idx + hyps_count + 1],
                    hyps_count + 1,
                    largest=False,
                )

                # Delete the worst hypothesis and update the worst score
                del self._beam_hyps[int((sorted_indices[0] + beam_idx))]
                del self._beam_scores[int((sorted_indices[0] + beam_idx))]
                self._beam_hyps_worst_scores[batch_idx] = sorted_next_scores[1]

            # We still have room for other hypotheses, so we just update the worst score
            else:
                # Update the worst score for the batch
                self._beam_hyps_worst_scores[batch_idx] = min(
                    score, self._beam_hyps_worst_scores[batch_idx]
                )
                self._beam_hyps_count[batch_idx] = hyps_count + 1

    def hypo_is_done(
        self, batch_idx: int, best_sum_logprobs: float, cur_len: int
    ) -> bool:
        """
        determines if hypothesis generation is complete for a batch. It first checks if there are enough hypotheses based on the beam size. 
        If early stopping is enabled, it returns True. Otherwise, it compares the score of the best possible hypothesis for the current step 
        (determined by best_sum_logprobs and cur_len) with the worst score among existing hypotheses for that batch. 
        If the worst existing hypothesis is better or equal, the method returns True, indicating completion. 
        Otherwise, it returns False, indicating that further generation is needed for the batch. 
        This process ensures that if new hypotheses are unlikely to surpass the worst ones, the batch is marked as done, optimizing efficiency in beam search.

        Parameters:
        -----------
        batch_idx : int
            Index of the batch.

        best_sum_logprobs : float
            Best sum of log probabilities for the batch.

        cur_len : int
            Current length of the hypotheses.

        Returns:
        --------
            bool: True if generation is done for the batch, else False.
        """
        # If there are not enough hypotheses, we are not done
        if self.hypo_len(batch_idx) < self.beam_size:
            return False

        # If early stopping is enabled, we are done as soon as there are enough hypotheses
        elif self.early_stopping:
            return True

        # If the best hypothesis we can generate is worse than the worst one in the heap, we are done
        else:
            cur_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self._beam_hyps_worst_scores[batch_idx].item() >= cur_score
            return ret

    def process(
        self,
        input_ids: torch.Tensor,
        next_scores: torch.Tensor,
        next_tokens: torch.Tensor,
        next_indices: torch.Tensor,
    ) -> tuple:
        """
        Process the next step in beam search.

        Parameters:
        -----------
        input_ids : torch.Tensor
            Input tokens.

        next_scores : torch.Tensor)
            Scores of the next tokens.

        next_tokens : torch.Tensor
            Next token candidates.

        next_indices : torch.Tensor
            Indices of the active beams.

        Returns:
        --------
            tuple: (next_beam_scores, next_beam_tokens, next_beam_indices)
        """
        # Input_ids: [batch_size * beam_size, cur_len]

        # Length of the hypothesis
        cur_len = input_ids.shape[-1]
        device = input_ids.device

        # Scores of the active beams
        next_beam_scores = torch.zeros(
            (self.batch_size, self.beam_size), dtype=next_scores.dtype, device=device
        )

        # The selected tokens for the active beams
        next_beam_tokens = torch.zeros(
            (self.batch_size, self.beam_size), dtype=next_tokens.dtype, device=device
        )

        # The indices of the active beams !
        next_beam_indices = torch.zeros(
            (self.batch_size, self.beam_size), dtype=next_indices.dtype, device=device
        )

        for batch_idx in range(self.batch_size):
            if self._done[batch_idx]:
                # If we are done with the batch we just pad the hypothesis
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = self.pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # Next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(
                    next_tokens[batch_idx],
                    next_scores[batch_idx],
                    next_indices[batch_idx],
                )
            ):
                # Since the inputs are of shape batch_size * beam_size. We need to find the batch_beam_idx.
                batch_beam_idx = batch_idx * self.beam_size + next_index

                if next_token == self.eos_token_id:
                    # If the EOS beam_token does not belong to top beam_size tokens, it should not be added

                    if beam_token_rank >= self.beam_size:
                        continue

                    self.hypo_add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                        batch_idx,
                    )

                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.beam_size:
                    break

            # Check if we have enough hypotheses for the batch and mark it as done.
            self._done[batch_idx] = self._done[batch_idx] or self.hypo_is_done(
                batch_idx,
                next_scores[batch_idx].max().item(),
                cur_len,
            )

        # Returns the scores, tokens and indices of the beams
        return (
            next_beam_scores.view(-1),
            next_beam_tokens.view(-1),
            next_beam_indices.view(-1),
        )

    def finalize(
        self,
        input_ids: torch.Tensor,
        final_beam_scores: torch.Tensor,
        final_beam_tokens: torch.Tensor,
        final_beam_indices: torch.Tensor,
        pad_token_id: int,
        eos_token_id: int,
    ) -> tuple:
        """
        Finalize beam search and return generated hypotheses.

        Parameters:
        -----------
            input_ids : torch.Tensor)
                Input tokens.

            final_beam_scores : torch.Tensor
                Scores of the final beams.

            final_beam_tokens : torch.Tensor
                Tokens of the final beams.

            final_beam_indices : torch.Tensor
                Indices of the final beams.

            pad_token_id : int
                ID of the padding token.

            eos_token_id : int
                ID of the end-of-sequence token.

        Returns:
            tuple: (decoded, best_scores)
        """

        batch_size = len(self._beam_hyps_count)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx in range(batch_size):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.beam_size):
                batch_beam_idx = batch_idx * self.beam_size + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                self.hypo_add(final_tokens, final_score, batch_idx)

        sent_lengths = torch.zeros(
            batch_size * self.num_hypo_to_return, dtype=torch.long
        )
        best = []
        best_scores = torch.zeros(
            batch_size * self.num_hypo_to_return,
            device=input_ids.device,
            dtype=torch.float32,
        )
        # retrieve best hypotheses
        for i in range(batch_size):
            # NOTE: lambda is not scriptable
            batch_hypo_start = (
                torch.sum(self._beam_hyps_count[:i])
                if i > 0
                else torch.tensor(0, dtype=torch.long)
            )
            batch_hypo_end = torch.sum(self._beam_hyps_count[: i + 1])
            beam_scores = torch.cat(self._beam_scores)[batch_hypo_start:batch_hypo_end]
            sorted_next_scores, sorted_indices = torch.topk(
                beam_scores, len(beam_scores), largest=True
            )

            for j in range(self.num_hypo_to_return):
                best_score = beam_scores[sorted_indices[j]]
                best_hyp = self._beam_hyps[batch_hypo_start + sorted_indices[j]]
                sent_lengths[self.num_hypo_to_return * i + j] = len(best_hyp)
                # append to lists
                best.append(best_hyp)
                best_scores[i * self.num_hypo_to_return + j] = best_score

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max() + 1, self.max_length)
        decoded = torch.zeros(
            batch_size * self.num_hypo_to_return, sent_max_len, dtype=torch.long
        )
        # shorter batches are padded if needed
        if sent_lengths.min() != sent_lengths.max():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < self.max_length:
                decoded[i, sent_lengths[i]] = eos_token_id

        return decoded, best_scores


def beam_search(
    init_hidden,
    init_cell,
    step_function,
    beam_scorer,
    batch_size,
    beam_size,
    max_length,
    vocab_size,
    bos_token_id,
    pad_token_id,
    eos_token_id,
):
    # Inputs of shape (1, batch_size * beam_size, cur_len)

    input_ids = torch.full(
        (batch_size * beam_size, 1),
        bos_token_id,
        dtype=torch.long,
        device=init_hidden.device,
    )
    hidden = init_hidden.repeat_interleave(beam_size, dim=1)
    cell = init_cell.repeat_interleave(beam_size, dim=1)

    beam_scores = torch.zeros(
        (batch_size, beam_size), dtype=torch.float, device=input_ids.device
    )

    # The first beam is only selected at first.
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * beam_size,))
    next_tokens = torch.zeros(
        (batch_size, beam_size), dtype=torch.long, device=input_ids.device
    )
    next_indices = torch.zeros(
        (batch_size, beam_size), dtype=torch.long, device=input_ids.device
    )

    cur_len = 1
    while cur_len < max_length:
        # Decoder forward step
        logits, hidden, cell = step_function(input_ids[..., -1], hidden, cell)
        next_token_scores = F.log_softmax(
            logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        # pre-process distribution

        ## Add beam scores to each token
        next_token_scores = next_token_scores + beam_scores.unsqueeze(1).expand_as(
            next_token_scores
        )

        # We select the top 2 * beam_size tokens
        next_token_scores = next_token_scores.view(batch_size, beam_size * vocab_size)
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * beam_size, dim=1, largest=True, sorted=True
        )

        # Scores of shape (batch_size, BEAM_SIZE * VOCAB_SIZE), we need to find from which beam the tokens come from and the actual token ids.
        next_indices = (
            next_tokens // vocab_size
        )  # This returns a tensor with values from 0 -> beam_size identifying the beam.
        next_tokens = (
            next_tokens % vocab_size
        )  # This returns a tensor with values from 0 -> vocab_size identifying the token.

        # This takes the input_ids, the scores and tokens of the next step. -> Beam_scores, choosen next tokens with their indices
        beam_scores, beam_next_tokens, beam_idx = beam_scorer.process(
            input_ids, next_token_scores, next_tokens, next_indices
        )

        input_ids = torch.cat(
            [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
        )
        hidden = hidden[:, beam_idx, :]
        cell = cell[:, beam_idx, :]

        if beam_scorer.is_done():
            break

        cur_len = cur_len + 1

    sequences, sequence_scores = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )

    return sequences, sequence_scores
