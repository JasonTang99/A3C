import gym
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

def train(rank, args, device, global_model, opt, opt_lock, scheduler,
          step_counter, max_reward, ma_reward, ma_loss):
    torch.manual_seed(args.seed + rank)

    # Setup env
    env = gym.make('SpaceInvaders-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    state = env.reset()
    next_state, _, _, _ = env.step(0)

    # Keep a deque of 4 processed states as input
    proc_state = transform(state, next_state)
    proc_states = deque([
        proc_state.clone(), proc_state.clone(),
        proc_state.clone(), proc_state.clone()
    ], maxlen=4)
    state = next_state

    # Setup local model
    local_model = ActorCritic([1, 4, 84, 84], action_size).to(device)

    thread_step_counter = 1
    ep_reward, ep_loss = 0.0, 0.0
    # Run an episode if the global maximum hasn't been reached
    while step_counter.value < args.steps:
        # Sync Models
        local_model.load_state_dict(global_model.state_dict())

        # Run simulation until episode done or update time
        values, rewards, actions, action_probs = [], [], [], []
        t_start = thread_step_counter
        done = False

        while not done and thread_step_counter - t_start != args.update_freq:
            # Take action according to policy
            processed_input = torch.cat(tuple(proc_states)).unsqueeze(0)
            logits, value = local_model(processed_input)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).detach()
            next_state, reward, done, _ = env.step(action)

            # Update states
            proc_state = transform(state, next_state)
            proc_states.append(proc_state)
            state = next_state

            # Store action probs, values, and rewards
            rewards.append(reward)
            values.append(value)
            actions.append(action)
            action_probs.append(probs)

            # Update counters
            step_counter.value += 1
            thread_step_counter += 1

        if args.verbose > 1 and step_counter.value % 100 == 0:
            print("{:.2f}%".format(100 * step_counter.value / args.steps))
        # env.render()

        rewards = torch.tensor(rewards).unsqueeze(1)  # (t, 1)
        values = torch.cat(values, 0)                 # (t, 1)
        actions = torch.cat(actions, 0)               # (t, 1)
        action_probs = torch.cat(action_probs, 0)     # (t, action_space)

        # Bootstrap the last state
        R = torch.zeros(1, 1)
        if not done:
            processed_input = torch.cat(tuple(proc_states)).unsqueeze(0)
            _, value = local_model(processed_input)
            R = value.detach()

        # Compute Returns
        returns = []
        for i in range(len(rewards))[::-1]:
            R = rewards[i] + args.gamma * R
            returns.extend(R)
        returns = torch.stack(returns[::-1])  # (t, 1)

        # Compute values needed in loss calculation
        t = action_probs.shape[0]
        log_probs = torch.log(action_probs)                     # (t, action_space)
        log_action_probs = torch.gather(log_probs, 1, actions)  # (t, 1)
        advantage = returns - values                            # (t, 1)
        entropy = -torch.sum(action_probs * log_probs) / t      # (1)

        # Calculate losses
        actor_loss = -torch.sum(log_action_probs *
                                advantage) - args.beta * entropy
        critic_loss = torch.sum(advantage.pow(2))
        total_loss = actor_loss + critic_loss

        # Calculate gradient and clip to maximum norm
        total_loss.backward()
        nn.utils.clip_grad_norm_(local_model.parameters(), args.max_grad)

        # Propogate gradients to shared model
        with opt_lock:
            for l_param, g_param in zip(local_model.parameters(), 
                                        global_model.parameters()):
                g_param._grad = l_param.grad.clone()
            opt.step()
            opt.zero_grad()

            # for group in opt.param_groups:
            #     print(group["lr"].value)
            # Anneal Learning Rate on first worker
            if rank == 0:
                for group in opt.param_groups:
                    group['lr'].value = scheduler.step()

        local_model.zero_grad()
        
        # Update episode metrics
        with torch.no_grad():
            ep_reward += torch.sum(rewards).item()
            ep_loss += total_loss.item()
        if done:
            with torch.no_grad():
                if ma_reward.value == 0.0:
                    ma_reward.value = ep_reward
                    ma_loss.value = ep_loss
                else:
                    ma_reward.value = ma_reward.value * 0.95 + ep_reward * 0.05
                    ma_loss.value = ma_loss.value * 0.95 + ep_loss * 0.05
                
                if max_reward.value < ma_reward.value and args.save_fp:
                    max_reward.value = ma_reward.value
                    if args.verbose > 1:
                        print("Saving new top model")
                        print("Saving new top model",
                              file=open('output', 'a'))
                        
                    # torch.save(
                    #     local_model.state_dict(), 
                    #     os.path.splitext(args.save_fp)[0] + "-best.pt"
                    # )
                    torch.save({
                        'model_state_dict': local_model.state_dict(),
                        # 'optimizer_state_dict': opt.state_dict(),
                    }, os.path.splitext(args.save_fp)[0] + "-ckt.tar")

                if args.verbose > 0:
                    s = f"MA Reward: {ma_reward.value:.2f}\t" + \
                        f"MA Loss: {ma_loss.value:.2f}\t" + \
                        f"EP Reward: {ep_reward}  \tEP Loss: {ep_loss:.4E}\t" + \
                        f"Thread-{rank} Steps: {thread_step_counter}"
                    print(s)
                    print(s, file=open('output', 'a'))

                ep_reward, ep_loss = 0.0, 0.0

                # Reset Environment
                state = env.reset()
                next_state, _, _, _ = env.step(0)

                # Keep a deque of 4 processed states as input
                proc_state = transform(state, next_state)
                proc_states = deque([
                    proc_state.clone(), proc_state.clone(),
                    proc_state.clone(), proc_state.clone()
                ], maxlen=4)
                state = next_state


def test(args, device, model, tries=3, steps=1000000):
    torch.manual_seed(args.seed)

    # Setup env
    env = gym.make('SpaceInvaders-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    # Run the simulation "tries" times
    for t in range(tries):
        # Reset Env
        state = env.reset()
        next_state, _, _, _ = env.step(0)

        # Keep a deque of 4 processed states as input
        proc_state = transform(state, next_state)
        proc_states = deque([
            proc_state.clone(), proc_state.clone(),
            proc_state.clone(), proc_state.clone()
        ], maxlen=4)
        state = next_state

        # Run simulation until episode done
        images = []
        ep_reward = 0.0
        for i in range(steps):
            if i % 3 == 0:
                env.render()
                screen = env.render(mode='rgb_array')
                images.append(Image.fromarray(screen))
            
            # Take action according to policy
            processed_input = torch.cat(tuple(proc_states)).unsqueeze(0)
            logits, value = model(processed_input)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).detach()
            next_state, reward, done, _ = env.step(action)

            # Update states
            proc_state = transform(state, next_state)
            proc_states.append(proc_state)
            state = next_state

            ep_reward += reward
            if done:
                break
        
        # Save as gif
        if args.verbose > 0:
            print(f'Try #{t} reward: {ep_reward}')
            print(f'Try #{t} reward: {ep_reward}', file=open('output', 'a'))
        images[0].save(f'invaders-{t}.gif', save_all=True,
                    append_images=images[1:], loop=0, duration=1)